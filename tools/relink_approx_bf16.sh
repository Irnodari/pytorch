#!/usr/bin/env bash
# Rebuild libapprox_bf16 and relink PyTorch's approximate bf16 GEMM against it.
#
# libapprox_bf16_device.a is device-linked into libtorch_approx_bf16_gemm.so, so a
# new arithmetic model only reaches the GPU after that library is relinked. The
# PyTorch build directory is persistent, so this relinks without recompiling.
#
# Usage: tools/relink_approx_bf16.sh [--skip-tests]
#   APPROX_BF16_SRC    approximate_fma checkout (default: ../approximate_fma)
#   APPROX_BF16_BUILD  its CMake build directory (default: $APPROX_BF16_SRC/build)
# Other build variables (TORCH_CUDA_ARCH_LIST, MAX_JOBS, CMAKE_ARGS, ...) pass through.

set -euo pipefail

pytorch_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
approx_src="${APPROX_BF16_SRC:-$pytorch_root/../approximate_fma}"
approx_build="${APPROX_BF16_BUILD:-$approx_src/build}"
run_tests=1
if [[ "${1:-}" == "--skip-tests" ]]; then
  run_tests=0
fi

if ! grep -q '^APPROX_BF16_ENABLE_CUDA:BOOL=ON' "$approx_build/CMakeCache.txt"; then
  echo "error: $approx_build is not configured with CUDA, so libapprox_bf16_device.a is not built." >&2
  echo "Reconfigure with: cmake -S $approx_src -B $approx_build -DAPPROX_BF16_ENABLE_CUDA=ON" >&2
  exit 1
fi

echo "==> Building libapprox_bf16 in $approx_build"
cmake --build "$approx_build" -j

prefix="$(sed -n 's/^CMAKE_INSTALL_PREFIX:PATH=//p' "$approx_build/CMakeCache.txt")"
echo "==> Installing libapprox_bf16 to $prefix"
if [[ -w "$prefix/lib" ]]; then
  cmake --install "$approx_build"
else
  sudo cmake --install "$approx_build"
fi

echo "==> Relinking libtorch_approx_bf16_gemm.so"
# Remove the outputs that embed the old device code, so they are relinked even if
# the build does not track the installed archive as a dependency.
find "$pytorch_root/build" -path '*torch_approx_bf16_gemm.dir*' -name 'cmake_device_link.o' -delete
rm -f "$pytorch_root"/build/lib/libtorch_approx_bf16_gemm.so*
cd "$pytorch_root"
USE_APPROX_BF16_GEMM=1 pip install -e . -v --no-build-isolation

if (( run_tests )); then
  echo "==> Checking the GPU results against the host model"
  python test/test_approx_bf16_gemm.py
fi
