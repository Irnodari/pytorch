# Approximate bf16 GEMM: environment variables

With `USE_APPROX_BF16_GEMM=1`, CUDA bf16 matrix multiplications run on CUTLASS SIMT
kernels whose arithmetic comes from `libapprox_bf16` (`../approximate_fma`), compiled
against the approximate CUTLASS in `../cutlass`. This file lists every environment
variable that controls it.

Only **bf16** is emulated. A model left in fp32 never touches these kernels: use
`model.cuda().bfloat16()` with bf16 inputs, or `torch.autocast("cuda", dtype=torch.bfloat16)`.

## Build time

Set these when running `pip install -e . -v --no-build-isolation`. They are stored in
the CMake cache, so changing one later needs a reconfigure (delete `build/CMakeCache.txt`).

| Variable | Default | Effect |
|---|---|---|
| `USE_APPROX_BF16_GEMM` | `0` | `1` builds `libtorch_approx_bf16_gemm.so` and routes bf16 GEMMs to it. Requires `USE_CUDA` on Linux. |
| `APPROX_BF16_ROOT` | `/usr/local` | Install prefix of `libapprox_bf16` (looks in `include/` and `lib/`). Leave it unset if the library is in a system path. |
| `APPROX_BF16_CUTLASS_DIR` | `../cutlass` | The approximate CUTLASS checkout. Header-only; never needs building. |
| `TORCH_CUDA_ARCH_LIST` | detected GPUs | Standard PyTorch variable. `libapprox_bf16_device.a` must contain code for **every** architecture listed here, or the device link fails. Its own list is `APPROX_BF16_CUDA_ARCHITECTURES` in `approximate_fma` (default `80;86;89;90`; `./build.sh --arch 120` for an RTX 50xx). |

## Run time

Each variable is read once per process, the first time PyTorch checks it, so set it
before starting Python.
Neither needs a rebuild.

| Variable | Default | Effect |
|---|---|---|
| `TORCH_APPROX_BF16_GEMM` | `1` | `0` turns the emulation off: bf16 GEMMs go back to cuBLAS/cuBLASLt, and conv, attention, RNN and grouped mm go back to their usual backends. Useful for exact-vs-approximate comparisons in the same build. Values other than `0`/`1` warn and leave it on. |
| `TORCH_APPROX_BF16_GEMM_ACC` | fp32 | `bf16` accumulates in bf16 instead of fp32. |

### Where the arithmetic comes from

The model is `src/bf16_model.h` in `approximate_fma`; `bf16_ops.cpp` only
exports it. Each output element accumulates over k in order, starting from 0,
with one multiply-add of the model per step:

| Mode | Each multiply-add | Final conversion to bf16 |
|---|---|---|
| fp32 accumulation (default) | `approx_bf16_fma_f32(a, b, acc)` | `approx_bf16_from_f32` |
| `TORCH_APPROX_BF16_GEMM_ACC=bf16` | `approx_bf16_fma(a, b, acc)` | `approx_bf16_from_f32` |

With bias or `beta = 1`, the epilogue adds C in fp32 before the final conversion.

The kernels don't call those functions for every multiply-add. The first bf16
GEMM on each GPU calls `approx_bf16_cuda_init()`, which uploads the model from
`libapprox_bf16.so`: the multiplier's products (`sig_mul_model`), the adder's
slice tables (`adder_slice_model`) and its configuration. The kernels then
run the model inline (`approx_bf16/cutlass_mma.h`). Each thread decodes its
tile's operands once, and each product is a lookup of the multiplier's product
followed by the FP unit. Tiles that contain a subnormal, inf or NaN operand, an
approximate adder with a bf16 accumulator, and models that change other stages
all run per element through `libapprox_bf16_device.a`. The results are
bit-identical in every case. Only the speed differs.

What a model change needs:

| You change in `bf16_model.h` | Then |
|---|---|
| `sig_mul_model`, `adder_slice_model` | rebuild and install `libapprox_bf16.so`, restart Python. No relink. |
| anything else | `tools/relink_approx_bf16.sh` |

The upload is synchronous, so it can't happen inside CUDA graph capture. Run one
bf16 GEMM on each GPU before capturing; the usual warm-up iteration does this.

### Checking the state from Python

```python
torch._C._cuda_isApproxBf16GemmEnabled()  # True when built with it and not disabled
```

## What goes through the emulated GEMM

When the emulation is on, these bf16 CUDA ops use it:

- `mm`, `addmm`, `bmm`, `baddbmm`, `matmul`, `linear`, and their backward passes.
- `conv1d/2d/3d` and transposed convs: cuDNN is skipped for bf16, so they use im2col + GEMM.
- `scaled_dot_product_attention` and `nn.MultiheadAttention`: the flash, memory-efficient
  and cuDNN backends are rejected, and the math backend keeps bf16 (it would normally
  convert to fp32).
- `nn.LSTM`/`GRU`/`RNN`: cuDNN is skipped, so the native cells' matmuls are used.
- `torch._grouped_mm`: one `mm` per group instead of the CUTLASS/cuBLASLt grouped kernels.

These are not emulated:

- Depthwise convs (they use a direct kernel, in exact fp32).
- 2:4 sparse semi-structured ops, `_mixed_dtypes_linear`, fp8 `_scaled_mm`.
- `torch.dot`.
- `torch.compile` Triton matmul templates (only with max-autotune).

## Relink script: `tools/relink_approx_bf16.sh`

After changing `src/bf16_model.h` (other than the two table models above), or
after updating `approximate_fma`, this rebuilds and installs `libapprox_bf16`,
relinks only `libtorch_approx_bf16_gemm.so`, and runs the tests. A header change
in `approximate_fma/include` (such as `cutlass_mma.h`) also needs
`ApproxBf16Gemm.cu` recompiled: delete its object file under `build/`, or touch
the source, before running the script. Build variables such as
`TORCH_CUDA_ARCH_LIST`, `MAX_JOBS` and `CMAKE_ARGS` pass through to the PyTorch build.

| Variable / flag | Default | Effect |
|---|---|---|
| `APPROX_BF16_SRC` | `../approximate_fma` | The `approximate_fma` checkout. |
| `APPROX_BF16_BUILD` | `$APPROX_BF16_SRC/build` | Its CMake build directory. Must be configured with `-DAPPROX_BF16_ENABLE_CUDA=ON`. |
| `--skip-tests` | off | Skip `test/test_approx_bf16_gemm.py` at the end. |

The library is installed to the `CMAKE_INSTALL_PREFIX` of `APPROX_BF16_BUILD`, using
`sudo` only when that directory is not writable.

## Tests: `test/test_approx_bf16_gemm.py`

The tests compare GPU results bit for bit with a sequential host loop over
`libapprox_bf16.so`, so they pass for any model, as long as the host and device
libraries are built from the same source. They skip unless the emulation is on.

- Run them with the same `TORCH_APPROX_BF16_GEMM_ACC` as your experiments, so the
  reference uses the same accumulator.
- The kernel-name test needs CUDA profiling (Kineto with CUPTI) and skips without it.
