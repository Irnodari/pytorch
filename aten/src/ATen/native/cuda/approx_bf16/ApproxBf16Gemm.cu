#include <ATen/native/cuda/approx_bf16/ApproxBf16Gemm.h>

#include <atomic>
#include <limits>
#include <mutex>

#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>

// Thread-tile multiply-add: each thread decodes its tile's operands once instead
// of once per multiply-add (bit-identical, several times faster). Must come
// before the GEMMs below are instantiated.
#include <approx_bf16/cutlass_mma.h>
#include <approx_bf16/ops.h>

#if !defined(CUTLASS_APPROX_BF16)
#error "ApproxBf16Gemm.cu must be compiled against the approximate CUTLASS with CUTLASS_APPROX_BF16"
#endif

namespace at::native::approx_bf16 {
namespace {

using bf16 = cutlass::bfloat16_t;
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

// SIMT only: tensor-core MMA multiplies bf16 in hardware and would bypass
// libapprox_bf16. Each thread accumulates over k in order, one rounding per
// step, with the model of approx_bf16_fma{,_f32}: on the inlined fast path of
// approx_bf16/cutlass_mma.h, or through cutlass::multiply_add, which
// CUTLASS_APPROX_BF16 maps to the library, for tiles with special values.
template <typename LayoutA, typename LayoutB, typename ElementC, typename ElementAccumulator>
using SimtGemm = cutlass::gemm::device::GemmUniversal<
    bf16, LayoutA,
    bf16, LayoutB,
    ElementC, ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

struct Problem {
  int m, n, k, num_batches;
  float alpha, beta;
  const void* a;
  int64_t lda, stridea;
  const void* b;
  int64_t ldb, strideb;
  void* c;
  int64_t ldc, stridec;
  cudaStream_t stream;
};

// Uploads the model's tables and configuration from libapprox_bf16.so to the
// current device, once per device. Without it every multiply-add would be a
// call into libapprox_bf16_device.a (same results, many times slower).
const char* ensure_model_on_device() {
  constexpr int kMaxDevices = 64;
  static std::atomic<bool> ready[kMaxDevices];
  static std::mutex mutex;
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return cudaGetErrorString(error);
  }
  if (device >= kMaxDevices) {
    return "device index too large for the approximate bf16 model";
  }
  if (ready[device].load(std::memory_order_acquire)) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex);
  if (!ready[device].load(std::memory_order_relaxed)) {
    error = cudaError_t(approx_bf16_cuda_init());
    if (error != cudaSuccess) {
      return cudaGetErrorString(error);
    }
    ready[device].store(true, std::memory_order_release);
  }
  return nullptr;
}

template <typename LayoutA, typename LayoutB, typename ElementC, typename ElementAccumulator>
const char* run(const Problem& p) {
  using Gemm = SimtGemm<LayoutA, LayoutB, ElementC, ElementAccumulator>;
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kBatched,
      {p.m, p.n, p.k},
      p.num_batches,
      {p.alpha, p.beta},
      p.a, p.b, p.c, p.c,
      p.stridea, p.strideb, p.stridec, p.stridec,
      p.lda, p.ldb, p.ldc, p.ldc);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status == cutlass::Status::kSuccess) {
    // Batched mode needs no workspace.
    status = gemm.initialize(args, nullptr, p.stream);
  }
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(p.stream);
  }
  return status == cutlass::Status::kSuccess ? nullptr : cutlassGetStatusString(status);
}

// A column-major m x k operand stored transposed is a row-major m x k operand
// with the same leading dimension.
template <typename ElementC, typename ElementAccumulator>
const char* dispatch_layouts(bool trans_a, bool trans_b, const Problem& p) {
  if (trans_a) {
    return trans_b ? run<RowMajor, RowMajor, ElementC, ElementAccumulator>(p)
                   : run<RowMajor, ColumnMajor, ElementC, ElementAccumulator>(p);
  }
  return trans_b ? run<ColumnMajor, RowMajor, ElementC, ElementAccumulator>(p)
                 : run<ColumnMajor, ColumnMajor, ElementC, ElementAccumulator>(p);
}

} // namespace

const char* bgemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const void* a,
    int64_t lda,
    int64_t stridea,
    const void* b,
    int64_t ldb,
    int64_t strideb,
    float beta,
    void* c,
    bool c_is_fp32,
    int64_t ldc,
    int64_t stridec,
    int64_t num_batches,
    bool bf16_accumulate,
    cudaStream_t stream) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  if (m > int_max || n > int_max || k > int_max || num_batches > int_max) {
    return "problem size does not fit in 32-bit integers";
  }
  if (const char* error = ensure_model_on_device()) {
    return error;
  }
  const Problem p{
      int(m), int(n), int(k), int(num_batches), alpha, beta,
      a, lda, stridea, b, ldb, strideb, c, ldc, stridec, stream};
  // 'c' (conjugate transpose) is the same as 't' for real operands.
  const bool trans_a = transa != 'n' && transa != 'N';
  const bool trans_b = transb != 'n' && transb != 'N';
  if (c_is_fp32) {
    return bf16_accumulate ? dispatch_layouts<float, bf16>(trans_a, trans_b, p)
                           : dispatch_layouts<float, float>(trans_a, trans_b, p);
  }
  return bf16_accumulate ? dispatch_layouts<bf16, bf16>(trans_a, trans_b, p)
                         : dispatch_layouts<bf16, float>(trans_a, trans_b, p);
}

} // namespace at::native::approx_bf16
