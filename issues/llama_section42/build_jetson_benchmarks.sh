#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$HERE/bin"
GGML_BUILD="$HERE/build-ggml-jetson"
CUTENSOR_ROOT=${POLYGEIST_CUTENSOR_ROOT:?set to an AArch64 cuTENSOR include/lib tree}
mkdir -p "$OUT"

DEFS=(-DMODEL_DIM=4096 -DFFN_DIM=11008 -DVOCAB=32000 -DSEQ_LEN=2048
      -DNUM_HEADS=32 -DWARMUP=5 -DMEASURE=5)
ARCH=(-march=armv8.2-a+fp16+bf16)

aarch64-linux-gnu-gcc -O3 "${ARCH[@]}" "${DEFS[@]}" \
  "$HERE/llama_extended_timing_harness.c" -lm \
  -o "$OUT/llama_native_cpu_7b_jetson"
aarch64-linux-gnu-gcc -O3 -ffast-math "${ARCH[@]}" "${DEFS[@]}" \
  "$HERE/llama_extended_timing_harness.c" -lm \
  -o "$OUT/llama_native_cpu_fastmath_7b_jetson"

PYTHON=/usr/bin/python3 \
POLYGEIST_CUTENSOR_ROOT="$CUTENSOR_ROOT" \
POLYGEIST_WRAP_KERNEL_PIPELINE=0 \
POLYGEIST_DISABLE_POINTWISE_MATCHING=1 \
POLYGEIST_DISABLED_KERNELS=cudaMaskSelect_f32_tensor,cudaAdd_f32_tensor,cudaSwiGLU_f32_tensor \
  "$ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
  --function=kernel_llama2_extended_forward \
  --harness="$HERE/llama_extended_timing_harness.c" \
  "$ROOT/third_party/cnn-extracted/llama2_extended_forward_bench.c" \
  "${DEFS[@]}" -o "$OUT/llama_polygeist_external_only_7b_jetson"

cmake -S "$ROOT/third_party/whisper.cpp" -B "$GGML_BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/aarch64-cuda-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
  -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP=OFF \
  -DGGML_CUDA_FA=OFF -DWHISPER_BUILD_EXAMPLES=OFF \
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_CUDA_FLAGS='-target-dir sbsa-linux'
cmake --build "$GGML_BUILD" -j2

CUDA_LIB=/usr/local/cuda/targets/sbsa-linux/lib
aarch64-linux-gnu-g++ -O3 -std=c++17 \
  -DMODEL_DIM=4096 -DFFN_DIM=11008 -DVOCAB=32000 -DSEQ_LEN=2048 \
  -DNUM_HEADS=32 "$ROOT/scripts/correctness/llama_extended_ggml_bench.cpp" \
  -I"$ROOT/third_party/whisper.cpp/ggml/include" \
  -L"$GGML_BUILD/ggml/src" -L"$GGML_BUILD/ggml/src/ggml-cuda" \
  -L"$CUDA_LIB" -L"$CUDA_LIB/stubs" -Wl,--no-as-needed \
  -lggml -lggml-base -lggml-cpu -lggml-cuda -lcublas -lcudart -lcuda \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath-link,"$CUDA_LIB" -lpthread -ldl -lm \
  -o "$OUT/llama_ggml_cuda_7b_jetson"
