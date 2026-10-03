#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

// Interface of libtorch_approx_bf16_gemm (built when USE_APPROX_BF16_GEMM=1).
//
// The library is compiled against the approximate CUTLASS (CUTLASS_APPROX_BF16),
// in which every cutlass::bfloat16_t multiply-add and conversion goes through
// libapprox_bf16. It is a separate shared library with hidden visibility so
// that its redefinition of cutlass::bfloat16_t cannot leak into the CUTLASS
// kernels compiled into libtorch_cuda, and because linking the device code of
// libapprox_bf16 requires -rdc, which is not viable for libtorch_cuda.

#define TORCH_APPROX_BF16_GEMM_API __attribute__((visibility("default")))

namespace at::native::approx_bf16 {

// cuBLAS-style column-major strided batched GEMM:
//   C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i]
// with bf16 A and B, and C either bf16 or fp32 (c_is_fp32). Products are
// accumulated in fp32 (approx_bf16_fma_f32), or in bf16 (approx_bf16_fma)
// when bf16_accumulate is set. Returns nullptr on success, otherwise a
// description of the error.
TORCH_APPROX_BF16_GEMM_API const char* bgemm(
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
    cudaStream_t stream);

} // namespace at::native::approx_bf16
