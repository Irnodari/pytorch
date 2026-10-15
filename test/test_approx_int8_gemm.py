# Owner(s): ["module: cuda"]

# Tests for builds with USE_APPROX_INT8_GEMM=1, where torch._int_mm and (with
# TORCH_APPROX_INT8_GEMM_FLOAT unset or 1) float / half / bf16 GEMMs run on the
# lookup-table int8 GEMM of libapprox_int8. Results are checked bit for bit
# against the library's host references, so they hold for any multiplier model
# (src/int8_model.h or APPROX_INT8_MUL_LUT).
#
# ATen computes a row-major out = mat1 @ mat2 as out^T = mat2^T @ mat1^T in
# column-major, so the multiplier's first operand comes from mat2 and its
# second from mat1; the references below do the same.

import ctypes
import math
import os
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


INT_MM_ENABLED = getattr(torch._C, "_cuda_isApproxInt8GemmEnabled", lambda: False)()
FLOAT_ENABLED = getattr(torch._C, "_cuda_isApproxInt8FloatGemmEnabled", lambda: False)()
BACKWARD_APPROX = os.environ.get("TORCH_APPROX_INT8_GEMM_BACKWARD") == "approx"

TYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def library():
    # torch_cuda already loaded it, so this is the same model.
    lib = ctypes.CDLL("libapprox_int8.so.1")
    i64, ptr = ctypes.c_int64, ctypes.c_void_p
    lib.approx_int8_gemm_host.argtypes = [ctypes.c_int, ctypes.c_int, i64, i64, i64,
                                          ptr, i64, i64, ptr, i64, i64, ptr, i64, i64, i64]
    lib.approx_int8_gemm_float_host.argtypes = [ctypes.c_int, ctypes.c_int, i64, i64, i64, ctypes.c_float,
                                                ctypes.c_int, ptr, i64, i64, ptr, i64, i64, ctypes.c_float,
                                                ctypes.c_int, ptr, i64, i64, i64]
    lib.approx_int8_model_name.restype = ctypes.c_char_p
    return lib


def multiplier_is_exact():
    """True if every product of the loaded multiplier is a * b."""
    lib = library()
    lib.approx_int8_mag_table.restype = ctypes.POINTER(ctypes.c_uint16)
    mag = lib.approx_int8_mag_table()
    return all(mag[a * 129 + b] == a * b for a in range(129) for b in range(129))


def reference_int_mm(x, y):
    """torch._int_mm(x, y) for int8 x (M x K), y (K x N)."""
    x, y = x.cpu().contiguous(), y.cpu().contiguous()
    (m, k), n = x.shape, y.shape[1]
    out = torch.empty(m, n, dtype=torch.int32)
    library().approx_int8_gemm_host(0, 0, n, m, k, y.data_ptr(), n, 0, x.data_ptr(), k, 0,
                                    out.data_ptr(), n, 0, 1)
    return out


def reference_mm(x, y, c=None, alpha=1.0, beta=0.0, out_dtype=None):
    """alpha * x @ y + beta * c for float x (... x M x K), y (... x K x N), batched over leading dims."""
    x, y = x.cpu().contiguous(), y.cpu().contiguous()
    m, k, n = x.shape[-2], x.shape[-1], y.shape[-1]
    batches = math.prod(x.shape[:-2])
    out_dtype = out_dtype or x.dtype
    shape = x.shape[:-1] + (n,)
    out = (c.cpu().to(out_dtype).expand(shape) if c is not None else torch.zeros(shape, dtype=out_dtype)).contiguous()
    library().approx_int8_gemm_float_host(0, 0, n, m, k, alpha, TYPES[x.dtype], y.data_ptr(), n, k * n,
                                          x.data_ptr(), k, m * k, beta, TYPES[out_dtype], out.data_ptr(), n,
                                          m * n, batches)
    return out


def assert_bitwise(test, got, want):
    got = got.cpu()
    test.assertEqual(got.dtype, want.dtype)
    test.assertEqual(got.shape, want.shape)
    same = (got == want) | (got.isnan() & want.isnan()) if got.is_floating_point() else got == want
    test.assertTrue(bool(same.all()), f"{int((~same).sum())} of {same.numel()} elements differ")


@unittest.skipIf(not INT_MM_ENABLED, "requires USE_APPROX_INT8_GEMM=1 and TORCH_APPROX_INT8_GEMM unset or 1")
class TestApproxInt8Gemm(TestCase):
    def _float(self, shape, device, dtype, transposed=False):
        if transposed:
            return torch.randn(shape[::-1], device=device, dtype=dtype).transpose(-1, -2)
        return torch.randn(shape, device=device, dtype=dtype)

    @onlyCUDA
    @parametrize("m,k,n", [(17, 8, 8), (32, 64, 16), (129, 136, 24)])
    def test_int_mm(self, device, m, k, n):
        x = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=device)
        y = torch.randint(-128, 128, (k, n), dtype=torch.int8, device=device)
        x[0, :] = -128  # magnitude 128: the kernel's general path
        assert_bitwise(self, torch._int_mm(x, y), reference_int_mm(x, y))

    @onlyCUDA
    @unittest.skipIf(not FLOAT_ENABLED, "TORCH_APPROX_INT8_GEMM_FLOAT=0")
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("m,k,n", [(1, 1, 1), (5, 33, 7), (130, 64, 129)])
    @parametrize("trans_a", [False, True])
    @parametrize("trans_b", [False, True])
    def test_mm(self, device, dtype, m, k, n, trans_a, trans_b):
        x = self._float((m, k), device, dtype, trans_a)
        y = self._float((k, n), device, dtype, trans_b)
        assert_bitwise(self, torch.mm(x, y), reference_mm(x, y))

    @onlyCUDA
    @unittest.skipIf(not FLOAT_ENABLED, "TORCH_APPROX_INT8_GEMM_FLOAT=0")
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_linear_with_bias(self, device, dtype):
        x = torch.randn(37, 48, device=device, dtype=dtype)
        w = torch.randn(20, 48, device=device, dtype=dtype)
        b = torch.randn(20, device=device, dtype=dtype)
        assert_bitwise(self, F.linear(x, w, b), reference_mm(x, w.t(), c=b, beta=1.0))

    @onlyCUDA
    @unittest.skipIf(not FLOAT_ENABLED, "TORCH_APPROX_INT8_GEMM_FLOAT=0")
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm(self, device, dtype):
        q = torch.randn(6, 40, 16, device=device, dtype=dtype)
        k = torch.randn(6, 16, 40, device=device, dtype=dtype)
        assert_bitwise(self, torch.bmm(q, k), reference_mm(q, k))

    @onlyCUDA
    @unittest.skipIf(not FLOAT_ENABLED, "TORCH_APPROX_INT8_GEMM_FLOAT=0")
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_attention_uses_math_backend(self, device, dtype):
        q, k, v = (torch.randn(2, 4, 32, 16, device=device, dtype=dtype) for _ in range(3))
        # The fused kernels refuse to run, so the default is the math backend,
        # whose matmuls are the emulated ones.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            with self.assertRaises(RuntimeError):
                F.scaled_dot_product_attention(q, k, v)
        with sdpa_kernel([SDPBackend.MATH]):
            want = F.scaled_dot_product_attention(q, k, v)
        assert_bitwise(self, F.scaled_dot_product_attention(q, k, v), want.cpu())

    @onlyCUDA
    @unittest.skipIf(not FLOAT_ENABLED or BACKWARD_APPROX, "needs float GEMMs, exact backward")
    def test_backward_is_exact(self, device):
        x = torch.randn(64, 96, device=device, requires_grad=True)
        w = torch.randn(96, 32, device=device, requires_grad=True)
        g = torch.randn(64, 32, device=device)
        y = x @ w
        y.backward(g)
        exact_y = x.detach().double() @ w.detach().double()
        exact_dx = g.double() @ w.detach().double().t()
        exact_dw = x.detach().double().t() @ g.double()
        # Forward: quantized (int8 error is ~1e-2 relative); backward: fp32.
        self.assertGreater(float((y.double() - exact_y).abs().max()), 1e-3)
        torch.testing.assert_close(x.grad.double(), exact_dx, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w.grad.double(), exact_dw, rtol=1e-4, atol=1e-4)

    @onlyCUDA
    @unittest.skipIf(not FLOAT_ENABLED, "TORCH_APPROX_INT8_GEMM_FLOAT=0")
    def test_conv2d_runs_through_gemm(self, device):
        x = torch.randn(2, 8, 12, 12, device=device)
        w = torch.randn(16, 8, 3, 3, device=device)
        got = F.conv2d(x, w, padding=1)
        want = F.conv2d(x.double(), w.double(), padding=1)  # fp64 is not emulated
        err = float((got.double() - want).abs().max() / want.abs().max())
        self.assertGreater(err, 1e-6)  # quantized ...
        # ... but close: int8 rounding alone stays within a few percent.  An
        # approximate multiplier adds its own error (the GEMM tests check it bit
        # for bit), so then only rule out garbage.
        self.assertLess(err, 5e-2 if multiplier_is_exact() else 1.0)


instantiate_device_type_tests(TestApproxInt8Gemm, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
