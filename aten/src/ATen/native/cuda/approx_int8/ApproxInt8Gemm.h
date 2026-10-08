#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

// Interface of libtorch_approx_int8_gemm (built when USE_APPROX_INT8_GEMM=1).
//
// The library runs int8 GEMMs on CUTLASS SIMT kernels over approx::aint8_t,
// whose multiply-accumulate comes from libapprox_int8 (../approximate_fma).
// It is a separate shared library with hidden visibility because linking the
// device code of libapprox_int8 requires -rdc, which is not viable for
// libtorch_cuda.

#define TORCH_APPROX_INT8_GEMM_API __attribute__((visibility("default")))

namespace at::native::approx_int8 {

// cuBLAS-style column-major GEMM with int8 A and B and an int32 C:
//   C = op(A) @ op(B)
// Each element of C accumulates over k in order, starting from 0, with
// approx_int8_mac. Returns nullptr on success, otherwise a description of the
// error.
TORCH_APPROX_INT8_GEMM_API const char* int8_gemm(
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
    cudaStream_t stream);

} // namespace at::native::approx_int8
