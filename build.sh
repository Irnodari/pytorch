#!/bin/bash

rm -rf build
USE_KINETO=0 CUDA_HOME=/usr/local/cuda-13.3 \
CMAKE_ARGS="-DCUDA_cupti_LIBRARY=/usr/local/cuda/extras/CUPTI/lib64/libcupti.so" \
MAX_JOBS=4 USE_APPROX_BF16_GEMM=1 USE_APPROX_INT8_GEMM=1 pip install -e . -v --no-build-isolation
