# Approximate arithmetic in this PyTorch fork

This fork can run CUDA matrix multiplications through emulated approximate
arithmetic circuits, from libraries built in `cutlass-approx-fma` (`../approximate_fma`
in some scripts):

| Build option | Library | What is emulated | Use it for |
|---|---|---|---|
| `USE_APPROX_INT8_GEMM=1` | `libapprox_int8` | int8 multiplications through an approximate 8x8 multiplier (lookup table); float / half / bf16 GEMMs are quantized to int8 on the fly | running and retraining int8-quantized networks, transformers included. **Start here.** |
| `USE_APPROX_BF16_GEMM=1` | `libapprox_bf16` + approximate CUTLASS | every bf16 fma through a bit-accurate multiplier / adder datapath model | studying bf16 arithmetic units; too slow for retraining |

Both are off unless built in, and each can be switched off at run time. The
[int8 part](#approximate-int8-gemm) is below the bf16 one.

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

# Approximate int8 GEMM

With `USE_APPROX_INT8_GEMM=1`, int8 GEMMs run on lookup-table kernels whose every
multiplication goes through an approximate 8x8 multiplier, from `libapprox_int8`
(built and installed with `libapprox_bf16` by `cutlass-approx-fma`). Two things go
through it:

- `torch._int_mm` (int8 x int8 -> int32): what torchao's int8 quantization calls in
  eager mode, so `quantize_(model, Int8DynamicActivationInt8WeightConfig())` uses it
  as is (`int8_dynamic_activation_int8_weight()` in older torchao).
- float, half and bf16 GEMMs, quantized to int8 on the fly: every row of one operand
  and every column of the other gets its own scale (amax / 127, round to nearest
  even, clamp to [-127, 127]); the int32 result is scaled back. Unmodified models
  then run every matmul, attention included, as an approximate int8 GEMM.

Signed operands use the unsigned multiplier on their magnitudes (sign-magnitude):
`a * b = sign(a) sign(b) mul8u(|a|, |b|)`. Products accumulate exactly in int32.

It can be built together with `USE_APPROX_BF16_GEMM`; when both are on, the int8
emulation takes bf16 GEMMs first (`TORCH_APPROX_INT8_GEMM_FLOAT=0` leaves them to the
bf16 one).

## Build time

| Variable | Default | Effect |
|---|---|---|
| `USE_APPROX_INT8_GEMM` | `0` | `1` links `libapprox_int8` into `libtorch_cuda` and routes the GEMMs above to it. Requires `USE_CUDA` on Linux. |
| `APPROX_INT8_ROOT` | `APPROX_BF16_ROOT`, else `/usr/local` | Install prefix of `libapprox_int8` (`include/approx_int8/ops.h`, `lib/libapprox_int8.so`). |

`libapprox_int8.so` must contain code for the GPUs it runs on
(`APPROX_BF16_CUDA_ARCHITECTURES` of `cutlass-approx-fma`, e.g. `./build.sh --arch 120`).

## Run time

Read once per process, so set them before starting Python. None needs a rebuild.

| Variable | Default | Effect |
|---|---|---|
| `TORCH_APPROX_INT8_GEMM` | `1` | `0` turns the emulation off: `_int_mm` goes back to cuBLASLt, float GEMMs to cuBLAS, and conv, attention and RNN to their usual backends. |
| `TORCH_APPROX_INT8_GEMM_FLOAT` | `1` | `0` emulates `torch._int_mm` only; float / half / bf16 GEMMs stay exact. |
| `TORCH_APPROX_INT8_GEMM_BACKWARD` | `exact` | `approx` quantizes the GEMMs of the backward pass too. By default they stay exact (cuBLAS), so the quantization acts as a straight-through estimator: the usual setting for retraining a network for an approximate multiplier. |
| `APPROX_INT8_GEMM_PATH` | `auto` | `libapprox_int8` runs multipliers whose table factors into few int8 features on the tensor cores (bit-identical, roughly one int8 GEMM per feature) and the others on its lookup kernel. `tensor` / `lookup` force one; see `cutlass-approx-fma`'s README ("Two GPU paths"). `APPROX_INT8_TC_MAX_FEATURES` (64) is the threshold of `auto`. |
| `APPROX_INT8_TC_WORKSPACE_MB` | `1024` | Workspace of the tensor-core path, from a memory pool of the library: outside PyTorch's caching allocator, so leave room for it. |
| `APPROX_INT8_MUL_LUT` | unset | A multiplier table file to use instead of the model compiled into `libapprox_int8` (`src/int8_model.h`): the `mul.hex` that `make_luts` / `make_luts_rtl` write, 65536 16-bit products at `(a << 8) \| b`. |

### Which operand goes to which multiplier input

ATen computes a row-major `out = mat1 @ mat2` as `out^T = mat2^T @ mat1^T` in column
major, so the multiplier's first input `a` gets the elements of `mat2` and its second
input `b` those of `mat1`. This matters for multipliers that are not symmetric:

| Op | `a` | `b` |
|---|---|---|
| `nn.Linear` (`x @ W^T`) | weight | activation |
| `torch._int_mm(x, w)` | `w` | `x` |
| attention `Q @ K^T` | key | query |
| attention `P @ V` | value | attention weights |
| `torch.mv(M, v)` | `v` | `M` |

(For outputs that are not contiguous, ATen may swap them.)

## Using it

```sh
# Once: build PyTorch with it (libapprox_int8 installed under APPROX_INT8_ROOT)
USE_APPROX_INT8_GEMM=1 pip install -e . -v --no-build-isolation

# Each run: pick the multiplier (a mul.hex from make_luts_rtl), or leave it unset
# for the model compiled into libapprox_int8
export APPROX_INT8_MUL_LUT=/path/to/luts/mul.hex
```

```python
import torch
assert torch._C._cuda_isApproxInt8FloatGemmEnabled()

# Inference: an unmodified model; every float / half / bf16 matmul is an
# approximate int8 GEMM (weights per output channel, activations per token).
model = MyModel().cuda().eval()          # fp32, or .bfloat16() / autocast
with torch.no_grad():
    out = model(x.cuda())

# Retraining for the approximate multiplier: train as usual. Forward GEMMs are
# approximate, backward GEMMs exact (straight-through estimator).
model.train()
for x, y in loader:
    loss = loss_fn(model(x.cuda()), y.cuda())
    loss.backward()
    opt.step()
    opt.zero_grad()
```

The exact baseline is the same script with `TORCH_APPROX_INT8_GEMM=0`.

Hugging Face transformers: load with `attn_implementation="sdpa"` (or `"eager"`),
never `"flash_attention_2"`, which calls the flash-attn package directly and bypasses
ATen. With `"sdpa"`, the fused kernels are rejected and the math backend runs the two
attention matmuls through the emulation.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda()
```

With torchao instead (inference only; its quantized weights are not trainable),
`quantize_(model, Int8DynamicActivationInt8WeightConfig())` quantizes the linear
layers its own way and runs them through `torch._int_mm`; attention still goes
through the float path. Run in eager mode: under `torch.compile`, inductor may
generate its own int8 kernels.

### Retraining

Nothing in the model changes: train as usual. The forward GEMMs are quantized and go
through the approximate multiplier; with the default exact backward, gradients are
those of the unquantized matmul at the same inputs. Activation checkpointing recomputes
the forward inside the backward pass, so recomputed GEMMs are exact there.

### Checking the state from Python

```python
torch._C._cuda_isApproxInt8GemmEnabled()       # _int_mm emulated
torch._C._cuda_isApproxInt8FloatGemmEnabled()  # float / half / bf16 GEMMs too
```

## What goes through the emulated GEMM

With float GEMMs on, for float32, float16 and bfloat16 CUDA tensors: `mm`, `addmm`,
`bmm`, `baddbmm`, `matmul`, `linear`, `mv` / `addmv`; `conv1d/2d/3d` and transposed
convs (cuDNN is skipped, so im2col + GEMM); `scaled_dot_product_attention` and
`nn.MultiheadAttention` (flash, memory-efficient and cuDNN attention are rejected; the
math backend's matmuls are emulated); `nn.LSTM`/`GRU`/`RNN` (native cells);
`torch._grouped_mm` (one `mm` per group).

Not emulated: float64; depthwise convs (direct kernel); `torch.dot`; fp8 `_scaled_mm`;
`torch.compile` Triton matmul templates (max-autotune only).

The first emulated GEMM on each GPU uploads the multiplier table synchronously, so it
can't happen during CUDA graph capture: run a warm-up iteration first.

## Changing the multiplier

`libapprox_int8` is linked like any shared library and reads its table at run time:
point `APPROX_INT8_MUL_LUT` at a new `mul.hex`, or edit `src/int8_model.h` and
rebuild and install `cutlass-approx-fma`. PyTorch is not rebuilt or relinked.

## Measuring the cost: `tools/bench_approx_int8.py`

Runs GEMMs (`_int_mm`, bf16 / fp16 `mm`, `linear`, `bmm`), attention and whole models
(a BERT-base encoder for inference and a training step, convs, ResNet-50 with
torchvision) twice, in separate processes: with `TORCH_APPROX_INT8_GEMM=0` (cuBLASLt
int8 and cuBLAS bf16 on the tensor cores, flash attention, cuDNN) and with it on.
Reports time, TOPS, the slowdown, and from one profiled call the share of GPU time in
tensor-core-looking kernels and in `libapprox_int8`'s kernels (`--kernels` lists them).

```sh
python tools/bench_approx_int8.py --json results.json   # --only 'gemm|sdpa' for a subset
python tools/bench_approx_int8.py --load results.json --kernels
```

`APPROX_INT8_MUL_LUT`, `APPROX_INT8_GEMM_PATH` and `TORCH_APPROX_INT8_GEMM_FLOAT` / `_BACKWARD`
pass through to the emulated run. On the tensor-core path of `libapprox_int8`, its cuBLASLt
int8 GEMMs count as tensor-core time and its expansion kernels as emulation time. `cutlass-approx-fma`'s `bench_int8_tc` compares the bare GEMMs, without PyTorch.

## Tests: `test/test_approx_int8_gemm.py`

Compares `_int_mm`, `mm`, `linear` with bias and `bmm` bit for bit with the host
references of `libapprox_int8` (any multiplier model), checks that attention uses the
math backend, that the backward pass is exact by default, and that conv2d goes through
the quantized GEMM. They skip unless the emulation is on.
