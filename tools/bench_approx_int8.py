"""Throughput lost to the approximate int8 emulation, against the exact kernels on tensor cores.

    python tools/bench_approx_int8.py                  # every workload, exact then approx
    python tools/bench_approx_int8.py --only 'gemm|sdpa' --kernels
    python tools/bench_approx_int8.py --json out.json  # save the results; --load out.json reprints them

Each workload runs twice, in separate processes (the settings are read once per process):

  exact   TORCH_APPROX_INT8_GEMM=0: cuBLASLt int8 (IMMA tensor cores) for torch._int_mm, cuBLAS /
          cuBLASLt bf16 / fp16 on tensor cores, flash attention, cuDNN convs: what the GPU does
          without the emulation.
  approx  TORCH_APPROX_INT8_GEMM=1: libapprox_int8 (lookup-table int8 GEMMs; float GEMMs quantized
          on the fly, attention on the math backend, convs as im2col + GEMM).

TORCH_APPROX_BF16_GEMM=0 in both, so a build with the bf16 emulation too doesn't slow the baseline.
Other settings (APPROX_INT8_MUL_LUT, APPROX_INT8_GEMM_PATH, TORCH_APPROX_INT8_GEMM_FLOAT,
TORCH_APPROX_INT8_GEMM_BACKWARD) pass through to the approx run.  The multiplier decides the speed:
libapprox_int8 runs it on the tensor cores if its table factors into few int8 features (the
"multiplier" line says how many), else on its lookup kernel.

Times are the mean over at least --time seconds of back-to-back calls after a warm-up, CUDA events
around them, so they include CPU launch overhead (which dominates small workloads).  TOPS counts
two operations per multiply-add of the workload's matmuls (the full product for attention).  One
more call runs under the profiler to split GPU time by kernel: "emu %" is libapprox_int8's own
kernels (lookup GEMM, amax scans, feature expansion and combination), "TC %" kernels whose names
look like tensor-core kernels (xmma, imma, hmma, tensorop, nvjet, flash, ...), including the
cuBLASLt int8 GEMMs of libapprox_int8's tensor-core path: a heuristic, check them with --kernels.
"""

import argparse
import ctypes
import gc
import json
import math
import os
import re
import subprocess
import sys


MARKER = "BENCH_APPROX_INT8_RESULT "
MODES = {
    "exact": {"TORCH_APPROX_INT8_GEMM": "0", "TORCH_APPROX_BF16_GEMM": "0"},
    "approx": {"TORCH_APPROX_INT8_GEMM": "1", "TORCH_APPROX_BF16_GEMM": "0"},
}
EMULATION_KERNEL = re.compile(r"(^|::)(gemm_kernel|amax_warp_per_line|amax_thread_per_line|expand_kernel|combine_kernel)<")
TENSOR_CORE_KERNEL = re.compile(
    r"xmma|imma|hmma|gmma|tensorop|nvjet|flash|fmha|s16816|h16816|i16832|i8816|bf16816|s1688|h1688", re.I
)


# ---------------------------------------------------------------------------
# Workloads: name -> builder returning (fn, multiply-adds).  Built in the child.
# ---------------------------------------------------------------------------

def workloads():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda"
    bf16 = torch.bfloat16

    def randn(*shape, dtype=bf16):
        return torch.randn(*shape, device=dev, dtype=dtype)

    def randint8(*shape):
        return torch.randint(-127, 128, shape, device=dev, dtype=torch.int8)

    def int_mm(m, n, k):
        # torch._int_mm(x, w.t()): x tokens x in, w out x in (torchao's layout, the TN GEMM of nn.Linear).
        def build():
            x, w = randint8(m, k), randint8(n, k)
            return (lambda: torch._int_mm(x, w.t())), m * n * k
        return build

    def mm(n, dtype):
        def build():
            a, b = randn(n, n, dtype=dtype), randn(n, n, dtype=dtype)
            return (lambda: a @ b), n**3
        return build

    def linear(tokens, d_in, d_out):
        def build():
            x, w, bias = randn(tokens, d_in), randn(d_out, d_in), randn(d_out)
            return (lambda: F.linear(x, w, bias)), tokens * d_in * d_out
        return build

    def bmm_qk(heads, s, d):
        def build():
            q, k = randn(heads, s, d), randn(heads, s, d)
            return (lambda: torch.bmm(q, k.transpose(1, 2))), heads * s * s * d
        return build

    def sdpa(b, h, s, d):
        def build():
            q, k, v = randn(b, h, s, d), randn(b, h, s, d), randn(b, h, s, d)
            return (lambda: F.scaled_dot_product_attention(q, k, v)), 2 * b * h * s * s * d
        return build

    def encoder(train, layers=12, b=8, s=512, d=768, heads=12, ff=3072):
        # BERT-base sized; dropout off so both modes run the same kernels besides the GEMMs.
        def build():
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True)
            model = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False).to(dev, bf16)
            x = randn(b, s, d)
            macs = layers * (b * s * (4 * d * d + 2 * d * ff) + 2 * b * s * s * d)
            if not train:
                model.eval()

                def fn():
                    with torch.no_grad():
                        return model(x)
                return fn, macs
            model.train()
            x.requires_grad_(True)

            def fn():
                model.zero_grad(set_to_none=True)
                model(x).float().square().mean().backward()
            return fn, 3 * macs  # forward + two GEMMs per GEMM in the backward pass
        return build

    def convnet(b=32, c=64, hw=56, layers=4):
        def build():
            mods = []
            for _ in range(layers):
                mods += [nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()]
            model = nn.Sequential(*mods).to(dev, bf16).eval()
            x = randn(b, c, hw, hw)

            def fn():
                with torch.no_grad():
                    return model(x)
            return fn, layers * b * hw * hw * c * c * 9
        return build

    def resnet50(b=64):
        def build():
            import torchvision  # noqa: F401: skipped (reported as an error) without torchvision

            model = torchvision.models.resnet50().to(dev, bf16).eval()
            x = randn(b, 3, 224, 224)

            def fn():
                with torch.no_grad():
                    return model(x)
            return fn, b * 4_089_184_256  # multiply-adds of one 224 x 224 image
        return build

    return {
        "gemm: _int_mm 4096^3": int_mm(4096, 4096, 4096),
        "gemm: _int_mm 2048 tok, 4096 -> 11008": int_mm(2048, 11008, 4096),
        "gemm: _int_mm 32 tok, 4096 -> 4096": int_mm(32, 4096, 4096),  # _int_mm needs m > 16
        "gemm: mm bf16 4096^3": mm(4096, bf16),
        "gemm: mm fp16 4096^3": mm(4096, torch.float16),
        "gemm: linear bf16 2048 tok, 4096 -> 11008": linear(2048, 4096, 11008),
        "gemm: linear bf16 16 tok, 4096 -> 4096": linear(16, 4096, 4096),
        "gemm: bmm bf16 QK^T 96 x 512 x 512 x 64": bmm_qk(96, 512, 64),
        "sdpa: bf16 b8 h12 s512 d64": sdpa(8, 12, 512, 64),
        "sdpa: bf16 b1 h32 s2048 d128": sdpa(1, 32, 2048, 128),
        "model: BERT-base encoder bf16, inference b8 s512": encoder(train=False),
        "model: BERT-base encoder bf16, train step b8 s512": encoder(train=True),
        "model: 4x conv3x3 64ch 56x56 bf16, inference b32": convnet(),
        "model: ResNet-50 bf16, inference b64 (torchvision)": resnet50(),
    }


# ---------------------------------------------------------------------------
# Child: runs every workload in one mode, prints the results as JSON.
# ---------------------------------------------------------------------------

def time_ms(fn, min_time):
    import torch

    for _ in range(3):  # warm-up: the first emulated GEMM uploads the multiplier table
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    stop.record()
    stop.synchronize()
    reps = int(min(max(math.ceil(min_time * 1e3 / max(start.elapsed_time(stop), 1e-3)), 3), 10000))
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / reps


def kernel_times(fn):
    """GPU time per kernel name (us) over one call, or None without CUDA profiling."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import profile, ProfilerActivity

    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
    except Exception as e:  # no Kineto / CUPTI
        print(f"  profiler unavailable: {e}", file=sys.stderr)
        return None
    totals = {}
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            totals[e.name] = totals.get(e.name, 0.0) + e.time_range.elapsed_us()
    return totals or None


def model_name():
    try:
        lib = ctypes.CDLL("libapprox_int8.so.1")
        lib.approx_int8_model_name.restype = ctypes.c_char_p
        return lib.approx_int8_model_name().decode()
    except OSError:
        return None


def child(mode, args):
    import torch

    int8_on = getattr(torch._C, "_cuda_isApproxInt8GemmEnabled", None)
    if int8_on is None:
        sys.exit("this torch has no approximate int8 GEMM (build the int8 branch with USE_APPROX_INT8_GEMM=1)")
    float_on = torch._C._cuda_isApproxInt8FloatGemmEnabled()
    bf16_on = getattr(torch._C, "_cuda_isApproxBf16GemmEnabled", lambda: False)()
    if (mode == "approx") != bool(int8_on()) or bf16_on:
        sys.exit(f"{mode}: unexpected state, int8 emulation {int8_on()}, bf16 emulation {bf16_on}")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")

    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    meta = {
        "gpu": f"{props.name} (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)",
        "torch": f"{torch.__version__} (git {torch.version.git_version[:12]}, CUDA {torch.version.cuda})",
        "int_mm emulated": bool(int8_on()),
        "float GEMMs emulated": bool(float_on),
        "backward": os.environ.get("TORCH_APPROX_INT8_GEMM_BACKWARD", "exact") if float_on else "-",
        "multiplier": model_name() if mode == "approx" else "exact (tensor cores)",
    }
    results = {}
    builders = workloads()
    for name, build in builders.items():
        if args.only and not re.search(args.only, name):
            continue
        print(f"[{mode}] {name}", file=sys.stderr, flush=True)
        try:
            fn, macs = build()
            ms = time_ms(fn, args.time)
            kernels = None if args.no_profile else kernel_times(fn)
            results[name] = {"ms": ms, "macs": macs, "kernels": kernels}
        except Exception as e:  # out of memory, missing torchvision, ...
            results[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}
            print(f"  {results[name]['error']}", file=sys.stderr)
        fn = None
        gc.collect()
        torch.cuda.empty_cache()
    print(MARKER + json.dumps({"meta": meta, "results": results}))


# ---------------------------------------------------------------------------
# Parent: runs the children, prints the comparison.
# ---------------------------------------------------------------------------

def run_mode(mode, args):
    env = dict(os.environ, **MODES[mode])
    cmd = [sys.executable, os.path.abspath(__file__), "--child", mode, "--time", str(args.time)]
    if args.only:
        cmd += ["--only", args.only]
    if args.no_profile:
        cmd.append("--no-profile")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
        print(line)
    sys.exit(f"the {mode} run failed (exit code {proc.returncode})")


def shares(kernels):
    """(emulation %, tensor-core %) of the GPU time, or None."""
    if not kernels:
        return None
    total = sum(kernels.values()) or 1.0
    emu = sum(t for k, t in kernels.items() if EMULATION_KERNEL.search(k))
    tc = sum(t for k, t in kernels.items() if TENSOR_CORE_KERNEL.search(k) and not EMULATION_KERNEL.search(k))
    return 100 * emu / total, 100 * tc / total


def report(data, show_kernels):
    modes = [m for m in ("exact", "approx") if m in data]
    for mode in modes:
        print(f"{mode}:")
        for k, v in data[mode]["meta"].items():
            print(f"  {k:22s} {v}")
    print()

    names = list(dict.fromkeys(n for m in modes for n in data[m]["results"]))
    width = max([len(n) for n in names] + [10])
    head = f"{'':{width}s}"
    for mode in modes:
        head += f" {mode + ' ms':>11s} {'TOPS':>7s} {'TC %':>5s} {'emu %':>5s}"
    if len(modes) == 2:
        head += f" {'slowdown':>9s}"
    print(head)
    for name in names:
        row = f"{name:{width}s}"
        ms = {}
        for mode in modes:
            r = data[mode]["results"].get(name)
            if r is None or "error" in r:
                row += f" {'error' if r else '-':>11s} {'':>7s} {'':>5s} {'':>5s}"
                continue
            ms[mode] = r["ms"]
            s = shares(r.get("kernels"))
            pct = f" {s[1]:5.0f} {s[0]:5.0f}" if s else f" {'?':>5s} {'?':>5s}"
            row += f" {r['ms']:11.3f} {2 * r['macs'] / r['ms'] * 1e-9:7.1f}{pct}"
        if len(ms) == 2:
            row += f" {ms['approx'] / ms['exact']:8.1f}x"
        print(row)

    errors = [(m, n, data[m]["results"][n]["error"]) for m in modes for n in names
              if "error" in data[m]["results"].get(n, {})]
    for mode, name, error in errors:
        print(f"  {mode} / {name}: {error}")

    if show_kernels:
        for name in names:
            print(f"\n{name}")
            for mode in modes:
                kernels = data[mode]["results"].get(name, {}).get("kernels")
                if not kernels:
                    continue
                total = sum(kernels.values()) or 1.0
                for k, t in sorted(kernels.items(), key=lambda kv: -kv[1])[:5]:
                    tag = "emu" if EMULATION_KERNEL.search(k) else "TC" if TENSOR_CORE_KERNEL.search(k) else ""
                    print(f"  {mode:6s} {100 * t / total:5.1f}% {tag:3s} {k[:150]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", help="regex: run the workloads whose name matches")
    p.add_argument("--time", type=float, default=1.0, help="seconds of repetitions per measurement (default 1)")
    p.add_argument("--modes", default="exact,approx", help="comma-separated subset of exact,approx")
    p.add_argument("--kernels", action="store_true", help="list the top kernels of each workload")
    p.add_argument("--no-profile", action="store_true", help="skip the profiler pass (no TC / emu shares)")
    p.add_argument("--json", help="save the results to this file")
    p.add_argument("--load", help="print results saved with --json instead of running")
    p.add_argument("--list", action="store_true", help="list the workloads")
    p.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:
        child(args.child, args)
        return
    if args.list:
        import torch  # noqa: F401: workloads() needs it

        print("\n".join(workloads()))
        return
    if args.load:
        with open(args.load) as f:
            data = json.load(f)
    else:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        if not modes or any(m not in MODES for m in modes):
            p.error("--modes: comma-separated subset of exact,approx")
        data = {mode: run_mode(mode, args) for mode in modes}
        if args.json:
            with open(args.json, "w") as f:
                json.dump(data, f, indent=1)
    report(data, args.kernels)
    if args.json and not args.load:
        print(f"\nresults saved to {args.json}")


if __name__ == "__main__":
    main()
