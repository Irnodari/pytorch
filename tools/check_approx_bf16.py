"""Checks whether bf16 matmuls run on the approximate CUTLASS GEMM, without the profiler.

    python tools/check_approx_bf16.py

Reports: which torch is imported and how it was configured, whether the emulation
is compiled in and enabled, whether libtorch_approx_bf16_gemm / libapprox_bf16 are
loaded, and for each op the time and whether its result differs from cuBLAS (the
same op rerun in a subprocess with TORCH_APPROX_BF16_GEMM=0). The emulated SIMT GEMM
is tens of times slower than cuBLAS on tensor cores, so the timing alone tells
the two apart.
"""

import os
import pathlib
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


N = 2048


def ops(device):
    g = torch.Generator(device="cpu").manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=torch.bfloat16)

    a, b, bias = rand(N, N), rand(N, N), rand(N)
    a3, b3 = rand(4, N // 2, N // 2), rand(4, N // 2, N // 2)
    a32, b32 = a.float(), b.float()

    def autocast_mm():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.mm(a32, b32)

    return {
        "mm": lambda: torch.mm(a, b),
        "matmul": lambda: a @ b,
        "linear+bias (addmm)": lambda: F.linear(a, b, bias),
        "bmm": lambda: torch.bmm(a3, b3),
        "autocast mm (fp32 inputs)": autocast_mm,
    }


def run_ops():
    """Times each op; returns {name: (ms, checksum of the result's bits)}."""
    results = {}
    for name, fn in ops("cuda").items():
        out = fn()
        torch.cuda.synchronize()
        reps = 3
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / reps
        bits = out.contiguous().view(torch.int16).to(torch.int64)
        checksum = int((bits * torch.arange(1, bits.numel() + 1, device=bits.device).view(bits.shape)).sum().item())
        results[name] = (ms, checksum)
    return results


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        for name, (ms, checksum) in run_ops().items():
            print(f"{name}\t{ms}\t{checksum}")
        return

    print(f"torch {torch.__version__} from {torch.__file__}")
    print(f"built with CUDA {torch.version.cuda}, git {torch.version.git_version[:12]}")
    root = pathlib.Path(torch.__file__).resolve().parent.parent
    cache = root / "build" / "CMakeCache.txt"
    if cache.exists():
        lines = [l.strip() for l in cache.read_text(errors="replace").splitlines() if "APPROX" in l and not l.startswith("//")]
        print(f"{cache}:")
        for l in lines or ["(no APPROX variables: configured without USE_APPROX_BF16_GEMM)"]:
            print(f"    {l}")
    else:
        print(f"no build directory next to this torch ({cache} missing): not the source build?")

    check = getattr(torch._C, "_cuda_isApproxBf16GemmEnabled", None)
    if check is None:
        print("PROBLEM: this torch has no _cuda_isApproxBf16GemmEnabled: it is not built from the patched tree")
        return
    enabled = check()
    print(f"_cuda_isApproxBf16GemmEnabled(): {enabled}")
    print(f"TORCH_APPROX_BF16_GEMM={os.environ.get('TORCH_APPROX_BF16_GEMM', '(unset)')}, "
          f"TORCH_APPROX_BF16_GEMM_ACC={os.environ.get('TORCH_APPROX_BF16_GEMM_ACC', '(unset: fp32)')}")
    if not enabled:
        print("PROBLEM: emulation off. Either the build has USE_APPROX_BF16_GEMM off (see the cache above;")
        print("  pip install -e . does not reconfigure an existing build/, delete build/CMakeCache.txt),")
        print("  or TORCH_APPROX_BF16_GEMM=0 is set.")

    approx = run_ops()
    maps = pathlib.Path("/proc/self/maps").read_text()
    libs = sorted({l.split()[-1] for l in maps.splitlines() if "approx_bf16" in l})
    print("loaded:", *(libs or ["(no approx_bf16 library mapped)"]), sep="\n    ")

    env = dict(os.environ, TORCH_APPROX_BF16_GEMM="0")
    child = subprocess.run([sys.executable, __file__, "--child"], env=env, capture_output=True, text=True)
    if child.returncode != 0:
        print("cuBLAS reference run failed:\n" + child.stderr)
        return
    cublas = {}
    for line in child.stdout.splitlines():
        name, ms, checksum = line.split("\t")
        cublas[name] = (float(ms), int(checksum))

    print(f"\n{'op (' + str(N) + '^3)':28} {'this process':>14} {'cuBLAS':>10}  result")
    for name, (ms, checksum) in approx.items():
        ref_ms, ref_checksum = cublas[name]
        same = "same bits as cuBLAS" if checksum == ref_checksum else "differs from cuBLAS"
        verdict = "emulated" if ms > 5 * ref_ms and checksum != ref_checksum else "NOT emulated (cuBLAS)"
        print(f"{name:28} {ms:11.3f} ms {ref_ms:7.3f} ms  {same}: {verdict}")


if __name__ == "__main__":
    main()
