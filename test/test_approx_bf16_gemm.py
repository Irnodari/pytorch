# Owner(s): ["module: cuda"]

# Tests for builds with USE_APPROX_BF16_GEMM=1, where CUDA bf16 GEMMs run on the
# approximate CUTLASS SIMT kernels. Results are checked bit for bit against a
# sequential host loop over libapprox_bf16.so, so they hold for any arithmetic
# model compiled into the library.

import ctypes
import os
import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


APPROX_ENABLED = getattr(torch._C, "_cuda_isApproxBf16GemmEnabled", lambda: False)()
BF16_ACCUMULATE = os.environ.get("TORCH_APPROX_BF16_GEMM_ACC") == "bf16"


def to_bits(t):
    return [x & 0xFFFF for x in t.cpu().contiguous().view(torch.int16).flatten().tolist()]


def from_bits(bits, shape):
    return torch.tensor([x - 0x10000 if x >= 0x8000 else x for x in bits], dtype=torch.int16).view(torch.bfloat16).reshape(shape)


def reference_mm(a, b, bias=None):
    # libtorch_approx_bf16_gemm already loaded the library, so this returns the same model.
    lib = ctypes.CDLL("libapprox_bf16.so.1")
    lib.approx_bf16_from_f32.restype = ctypes.c_uint16
    lib.approx_bf16_from_f32.argtypes = [ctypes.c_float]
    lib.approx_bf16_fma.restype = ctypes.c_uint16
    lib.approx_bf16_fma.argtypes = [ctypes.c_uint16] * 3
    lib.approx_bf16_fma_f32.restype = ctypes.c_float
    lib.approx_bf16_fma_f32.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_float]

    (m, k), n = a.shape, b.shape[1]
    a_bits, b_bits = to_bits(a), to_bits(b)
    bias_f = bias.float().cpu() if bias is not None else torch.zeros(n)
    out = []
    for i in range(m):
        for j in range(n):
            if BF16_ACCUMULATE:
                acc = 0
                for p in range(k):
                    acc = lib.approx_bf16_fma(a_bits[i * k + p], b_bits[p * n + j], acc)
                acc = from_bits([acc], ()).item()
            else:
                acc = 0.0
                for p in range(k):
                    acc = lib.approx_bf16_fma_f32(a_bits[i * k + p], b_bits[p * n + j], acc)
            # The epilogue computes alpha * acc + beta * C in fp32 with alpha = beta = 1.
            out.append(lib.approx_bf16_from_f32((torch.tensor(acc) + bias_f[j]).item()))
    return from_bits(out, (m, n))


@unittest.skipIf(not APPROX_ENABLED, "requires USE_APPROX_BF16_GEMM=1 and TORCH_APPROX_BF16_GEMM unset or 1")
class TestApproxBf16Gemm(TestCase):
    def _make(self, shape, device, transposed):
        if transposed:
            return torch.randn(shape[::-1], device=device, dtype=torch.bfloat16).t()
        return torch.randn(shape, device=device, dtype=torch.bfloat16)

    @onlyCUDA
    @parametrize("m,k,n", [(1, 1, 1), (5, 33, 7), (17, 64, 3), (1, 40, 9)])
    @parametrize("trans_a", [False, True])
    @parametrize("trans_b", [False, True])
    def test_mm_matches_host_model(self, device, m, k, n, trans_a, trans_b):
        a = self._make((m, k), device, trans_a)
        b = self._make((k, n), device, trans_b)
        self.assertEqual(torch.mm(a, b).cpu(), reference_mm(a, b), atol=0, rtol=0)

    @onlyCUDA
    def test_linear_with_bias_matches_host_model(self, device):
        x = torch.randn(6, 19, device=device, dtype=torch.bfloat16)
        w = torch.randn(5, 19, device=device, dtype=torch.bfloat16)
        bias = torch.randn(5, device=device, dtype=torch.bfloat16)
        self.assertEqual(F.linear(x, w, bias).cpu(), reference_mm(x, w.t(), bias.cpu()), atol=0, rtol=0)

    @onlyCUDA
    def test_bmm_matches_host_model(self, device):
        a = torch.randn(3, 4, 21, device=device, dtype=torch.bfloat16)
        b = torch.randn(3, 21, 6, device=device, dtype=torch.bfloat16)
        expected = torch.stack([reference_mm(a[i], b[i]) for i in range(3)])
        self.assertEqual(torch.bmm(a, b).cpu(), expected, atol=0, rtol=0)

    @onlyCUDA
    @parametrize(
        "op",
        ["mm", "bmm", "linear", "linear_backward", "conv2d", "conv2d_backward", "sdpa", "sdpa_backward", "grouped_mm", "lstm"],
    )
    def test_ops_run_cutlass_kernel(self, device, op):
        def make(*shape):
            return torch.randn(*shape, device=device, dtype=torch.bfloat16, requires_grad=op.endswith("_backward"))

        fns = {
            "mm": lambda: torch.mm(make(32, 16), make(16, 8)),
            "bmm": lambda: torch.bmm(make(2, 32, 16), make(2, 16, 8)),
            "linear": lambda: F.linear(make(32, 16), make(8, 16), make(8)),
            "linear_backward": lambda: F.linear(make(32, 16), make(8, 16), make(8)).sum().backward(),
            "conv2d": lambda: F.conv2d(make(2, 3, 8, 8), make(4, 3, 3, 3), make(4)),
            "conv2d_backward": lambda: F.conv2d(make(2, 3, 8, 8), make(4, 3, 3, 3), make(4)).sum().backward(),
            "sdpa": lambda: F.scaled_dot_product_attention(make(2, 4, 16, 32), make(2, 4, 16, 32), make(2, 4, 16, 32)),
            "sdpa_backward": lambda: F.scaled_dot_product_attention(
                make(2, 4, 16, 32), make(2, 4, 16, 32), make(2, 4, 16, 32), is_causal=True
            ).sum().backward(),
            "grouped_mm": lambda: torch._grouped_mm(
                make(32, 16), make(2, 16, 8), offs=torch.tensor([16, 32], device=device, dtype=torch.int32)
            ),
            "lstm": lambda: torch.nn.LSTM(16, 8, device=device, dtype=torch.bfloat16)(make(5, 2, 16)),
        }
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fns[op]()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events()]
        self.assertTrue(any("GemmUniversal" in name for name in names), f"no CUTLASS GEMM kernel among {names}")
        # Fused attention and cuDNN kernels multiply on tensor cores.
        tensor_core_kernels = [name for name in names if any(k in name for k in ("flash_fwd", "flash_bwd", "fmha", "cudnn"))]
        self.assertEqual(tensor_core_kernels, [])


instantiate_device_type_tests(TestApproxBf16Gemm, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
