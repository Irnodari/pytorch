# Owner(s): ["module: cuda"]

# Tests for builds with USE_APPROX_INT8_GEMM=1, where CUDA int8 GEMMs
# (torch._int_mm) run on CUTLASS SIMT kernels whose multiply-accumulate is the
# model of libapprox_int8 (../approximate_fma). Results are checked bit for bit
# against a sequential host loop over libapprox_int8.so, so they hold for any
# arithmetic model compiled into the library or loaded from table files.
#
# test_table_files_reach_gpu also checks the GPU against an independent Python
# model of an approximate multiplier and adder whose tables it writes itself,
# in a subprocess (the tables are read once per process).

import ctypes
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


APPROX_ENABLED = getattr(torch._C, "_cuda_isApproxInt8GemmEnabled", lambda: False)()
CUDA_PROFILING = torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()


def host_library():
    # libtorch_approx_int8_gemm already loaded the library, so this returns the same model.
    lib = ctypes.CDLL("libapprox_int8.so.1")
    lib.approx_int8_mac.restype = ctypes.c_int32
    lib.approx_int8_mac.argtypes = [ctypes.c_int8, ctypes.c_int8, ctypes.c_int32]
    lib.approx_int8_model_name.restype = ctypes.c_char_p
    lib.approx_int8_mul_exact.restype = ctypes.c_int
    lib.approx_int8_add_exact_mask.restype = ctypes.c_uint32
    return lib


def reference_int_mm(a, b):
    """a @ b as the kernel computes it: each output accumulates over k in order from 0."""
    mac = host_library().approx_int8_mac
    (m, k), n = a.shape, b.shape[1]
    a_rows = a.cpu().tolist()
    b_cols = b.cpu().t().tolist()
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        row = a_rows[i]
        for j in range(n):
            col = b_cols[j]
            acc = 0
            for p in range(k):
                acc = mac(row[p], col[p], acc)
            out[i][j] = acc
    return torch.tensor(out, dtype=torch.int32)


def model_is_exact():
    lib = host_library()
    return bool(lib.approx_int8_mul_exact()) and lib.approx_int8_add_exact_mask() == 0xF


# An approximate multiplier and adder, for test_table_files_reach_gpu. The same
# circuits as `make_int8_luts --demo loa` (tools/int8_circuits.h in approximate_fma).
CIRCUITS = textwrap.dedent(
    """
    def truncated_mul(a, b):
        # Exact product with its 4 low bits cleared.
        return (a * b) & ~15

    def loa_add(x, y):
        # 32-bit lower-part OR adder: bits 0..5 are x | y, carry x5 & y5 into the exact upper part.
        carry = (x >> 5) & (y >> 5) & 1
        return (((x | y) & 63) | (((x >> 6) + (y >> 6) + carry) << 6)) & 0xFFFFFFFF

    def to_int32(v):
        return v - (1 << 32) if v & 0x80000000 else v

    def model_int_mm(a, b):
        # Sequential accumulation from 0, with the product sign-extended to 32 bits.
        m, k, n = len(a), len(a[0]), len(b[0])
        out = []
        for i in range(m):
            row = []
            for j in range(n):
                acc = 0
                for p in range(k):
                    acc = loa_add(acc, truncated_mul(a[i][p], b[p][j]) & 0xFFFFFFFF)
                row.append(to_int32(acc))
            out.append(row)
        return out
    """
)


def write_tables(directory):
    """Writes the CIRCUITS' tables in the libapprox_int8 file format, returns the env vars."""
    scope = {}
    exec(CIRCUITS, scope)
    mul_path = os.path.join(directory, "int8_mul.hex")
    add_path = os.path.join(directory, "int8_add.hex")
    with open(mul_path, "w") as f:
        f.write("// index ((uint8_t)a << 8) | (uint8_t)b, 16-bit product\n")
        for ua in range(256):
            a = ua - 256 if ua >= 128 else ua
            for ub in range(256):
                b = ub - 256 if ub >= 128 else ub
                f.write("%04x\n" % (scope["truncated_mul"](a, b) & 0xFFFF))
    with open(add_path, "w") as f:
        f.write("// 4 slices, index (cin << 16) | (x << 8) | y, value sum | carry_out << 8\n")
        for i in range(4):
            lines = []
            for cin in range(2):
                for x in range(256):
                    for y in range(256):
                        if i == 0:  # the OR part is inside slice 0; the adder's carry-in is unused
                            s = ((x | y) & 63) | (((x >> 6) + (y >> 6) + ((x >> 5) & (y >> 5) & 1)) << 6)
                        else:
                            s = x + y + cin
                        lines.append("%03x" % (s & 0x1FF))
            f.write("\n".join(lines) + "\n")
    return {"APPROX_INT8_MUL_LUT": mul_path, "APPROX_INT8_ADD_LUT": add_path}


def run_python(code, env_updates):
    env = dict(os.environ)
    env.update(env_updates)
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )


@unittest.skipIf(not APPROX_ENABLED, "requires USE_APPROX_INT8_GEMM=1 and TORCH_APPROX_INT8_GEMM unset or 1")
class TestApproxInt8Gemm(TestCase):
    def _make(self, shape, device, transposed, low=-128, high=128):
        if transposed:
            return torch.randint(low, high, shape[::-1], device=device, dtype=torch.int8).t()
        return torch.randint(low, high, shape, device=device, dtype=torch.int8)

    @onlyCUDA
    @parametrize("m,k,n", [(1, 1, 1), (5, 33, 7), (17, 64, 8), (24, 40, 16), (70, 9, 66)])
    @parametrize("trans_a", [False, True])
    @parametrize("trans_b", [False, True])
    def test_int_mm_matches_host_model(self, device, m, k, n, trans_a, trans_b):
        # Shapes cuBLASLt would reject (m <= 16, k or n not multiples of 8) are fine here.
        torch.manual_seed(m * 1000 + k * 10 + n)
        a = self._make((m, k), device, trans_a)
        b = self._make((k, n), device, trans_b)
        self.assertEqual(torch._int_mm(a, b).cpu(), reference_int_mm(a, b), atol=0, rtol=0)

    @onlyCUDA
    def test_int_mm_out_matches_host_model(self, device):
        a = self._make((32, 24), device, False)
        b = self._make((24, 16), device, False)
        out = torch.empty(32, 16, device=device, dtype=torch.int32)
        torch._int_mm(a, b, out=out)
        self.assertEqual(out.cpu(), reference_int_mm(a, b), atol=0, rtol=0)

    @onlyCUDA
    def test_int32_wraparound_matches_host_model(self, device):
        # 140000 * (-128 * -128) > 2^31: the accumulator wraps, like the hardware's.
        k = 140000
        a = torch.full((1, k), -128, device=device, dtype=torch.int8)
        b = torch.full((k, 2), -128, device=device, dtype=torch.int8)
        b[::3, 1] = 127
        self.assertEqual(torch._int_mm(a, b).cpu(), reference_int_mm(a, b), atol=0, rtol=0)

    @onlyCUDA
    def test_matches_cpu_int_mm_when_model_is_exact(self, device):
        if not model_is_exact():
            self.skipTest("the library's model is approximate")
        a = self._make((48, 64), device, False)
        b = self._make((64, 40), device, True)
        self.assertEqual(torch._int_mm(a, b).cpu(), torch._int_mm(a.cpu(), b.cpu()), atol=0, rtol=0)

    @onlyCUDA
    def test_table_files_reach_gpu(self, device):
        # A fresh process loads the tables written here through APPROX_INT8_*_LUT; its GPU
        # results must match an independent Python model of the same circuits, and differ
        # from exact arithmetic.
        code = CIRCUITS + textwrap.dedent(
            """
            import torch
            torch.manual_seed(0)
            a = torch.randint(-128, 128, (19, 37), dtype=torch.int8)
            b = torch.randint(-128, 128, (37, 11), dtype=torch.int8)
            got = torch._int_mm(a.cuda(), b.cuda()).cpu().tolist()
            want = model_int_mm(a.tolist(), b.tolist())
            exact = (a.int() @ b.int()).tolist()
            bad = sum(g != w for gr, wr in zip(got, want) for g, w in zip(gr, wr))
            differ = sum(g != e for gr, er in zip(got, exact) for g, e in zip(gr, er))
            print("mismatches", bad, "differ_from_exact", differ)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_python(code, write_tables(directory))
        self.assertEqual(result.returncode, 0, result.stderr)
        words = result.stdout.split()
        self.assertEqual(words[0:2], ["mismatches", "0"], result.stdout + result.stderr)
        self.assertGreater(int(words[3]), 0, "the approximate model gave exact results: tables not used?")

    @onlyCUDA
    def test_disabled_by_env(self, device):
        # TORCH_APPROX_INT8_GEMM=0 restores cuBLASLt (exact, with its shape restrictions).
        code = textwrap.dedent(
            """
            import torch
            assert not torch._C._cuda_isApproxInt8GemmEnabled()
            a = torch.randint(-128, 128, (32, 64), dtype=torch.int8)
            b = torch.randint(-128, 128, (64, 16), dtype=torch.int8)
            assert torch.equal(torch._int_mm(a.cuda(), b.cuda()).cpu(), a.int() @ b.int())
            try:
                torch._int_mm(a[:5].cuda(), b.cuda())
            except RuntimeError:
                print("ok")
            """
        )
        result = run_python(code, {"TORCH_APPROX_INT8_GEMM": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok", result.stdout + result.stderr)

    @onlyCUDA
    @unittest.skipIf(not CUDA_PROFILING, "requires a build with Kineto and CUPTI")
    def test_int_mm_runs_cutlass_kernel(self, device):
        a = self._make((32, 16), device, False)
        b = self._make((16, 8), device, False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            torch._int_mm(a, b)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events()]
        self.assertTrue(any("GemmUniversal" in name for name in names), f"no CUTLASS GEMM kernel among {names}")
        self.assertFalse(any("cublas" in name.lower() for name in names), f"a cuBLAS kernel ran: {names}")


instantiate_device_type_tests(TestApproxInt8Gemm, globals(), only_for="cuda")

if __name__ == "__main__":
    if APPROX_ENABLED:
        print("libapprox_int8 model:", host_library().approx_int8_model_name().decode())
    run_tests()
