#include <ATen/native/cuda/approx_int8/ApproxInt8Gemm.h>

#include <atomic>
#include <limits>
#include <mutex>

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>

// multiply_add<aint8_t, aint8_t, int32_t> and the thread-tile multiply-
// accumulate (approx_int8/cutlass_mma.h). Must come before the GEMMs below are
// instantiated.
#include <approx_int8/cutlass_support.h>
#include <approx_int8/ops.h>

namespace at::native::approx_int8 {
namespace {

using A8 = approx::aint8_t;
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

// SIMT only: dp4a and IMMA tensor cores multiply int8 in hardware and would
// bypass libapprox_int8. Each thread accumulates over k in order, starting from
// 0, with the model of approx_int8_mac. The epilogue stays in int32 (alpha = 1,
// beta = 0), so the accumulators are stored unchanged.
template <typename LayoutA, typename LayoutB>
using SimtGemm = cutlass::gemm::device::GemmUniversal<
    A8, LayoutA,
    A8, LayoutB,
    int32_t, ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

struct Problem {
  int m, n, k;
  const int8_t* a;
  int64_t lda;
  const int8_t* b;
  int64_t ldb;
  int32_t* c;
  int64_t ldc;
  cudaStream_t stream;
};

// Uploads the model's tables and configuration from libapprox_int8.so to the
// current device, once per device. Without it every multiply-accumulate would
// be a call into libapprox_int8_device.a running the model compiled into it,
// which ignores APPROX_INT8_MUL_LUT / APPROX_INT8_ADD_LUT.
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
    return "device index too large for the approximate int8 model";
  }
  if (ready[device].load(std::memory_order_acquire)) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex);
  if (!ready[device].load(std::memory_order_relaxed)) {
    error = cudaError_t(approx_int8_cuda_init());
    if (error != cudaSuccess) {
      return cudaGetErrorString(error);
    }
    ready[device].store(true, std::memory_order_release);
  }
  return nullptr;
}

template <typename LayoutA, typename LayoutB>
const char* run(const Problem& p) {
  using Gemm = SimtGemm<LayoutA, LayoutB>;
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {p.m, p.n, p.k},
      1,
      {1, 0},
      p.a, p.b, p.c, p.c,
      0, 0, 0, 0,
      p.lda, p.ldb, p.ldc, p.ldc);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status == cutlass::Status::kSuccess) {
    // One k partition (split-k would add partial sums with the exact adder),
    // so no workspace.
    status = gemm.initialize(args, nullptr, p.stream);
  }
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(p.stream);
  }
  if (status != cutlass::Status::kSuccess) {
    return cutlassGetStatusString(status);
  }
  cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}

} // namespace

const char* int8_gemm(
    bool transpose_a,
    bool transpose_b,
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* a,
    int64_t lda,
    const int8_t* b,
    int64_t ldb,
    int32_t* c,
    int64_t ldc,
    cudaStream_t stream) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  if (m > int_max || n > int_max || k > int_max) {
    return "problem size does not fit in 32-bit integers";
  }
  if (const char* error = ensure_model_on_device()) {
    return error;
  }
  const Problem p{int(m), int(n), int(k), a, lda, b, ldb, c, ldc, stream};
  // A column-major m x k operand stored transposed is a row-major m x k operand
  // with the same leading dimension.
  if (transpose_a) {
    return transpose_b ? run<RowMajor, RowMajor>(p) : run<RowMajor, ColumnMajor>(p);
  }
  return transpose_b ? run<ColumnMajor, RowMajor>(p) : run<ColumnMajor, ColumnMajor>(p);
}

} // namespace at::native::approx_int8
