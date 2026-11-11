#!/usr/bin/env bash
# Build the paper's one-token, one-layer FP32 Llama fixture from source.
# The session harness owns disjoint arrays, permitting argument residency.
# No retained compiler objects are used. See issues/llama_section42/README.md.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common_env.sh"
OUT=${1:?usage: build_llama_reproduction.sh OUTPUT_DIRECTORY}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BENCH="$REPO_ROOT/issues/llama_section42"
SOURCE="$REPO_ROOT/third_party/cnn-extracted/llama2_extended_forward_bench.c"
CUDA=${CUDA:-/usr/local/cuda-12.6/targets/sbsa-linux}
GGML_BUILD=${POLYGEIST_GGML_BUILD:-$BENCH/build-ggml-jetson}
: "${POLYGEIST_CUTENSOR_ROOT:?set to the AArch64 cuTENSOR tree}"
DEFS=(-DMODEL_DIM=4096 -DFFN_DIM=11008 -DVOCAB=32000 -DSEQ_LEN=2048
      -DNUM_HEADS=32 -DDUMP_ALL=1)
CC=${AARCH64_CC:-aarch64-linux-gnu-gcc}
CXX=${AARCH64_CXX:-aarch64-linux-gnu-g++}
"$CC" -O3 -march=armv8.2-a+fp16+bf16 "${DEFS[@]}" -DWARMUP=5 -DMEASURE=5 \
  -DLLAMA_HARNESS_NATIVE "$BENCH/llama_extended_timing_harness.c" -lm \
  -o "$OUT/llama_native"
POLYGEIST_WRAP_KERNEL_PIPELINE=0 POLYGEIST_DISABLE_POINTWISE_MATCHING=0 \
POLYGEIST_KEEP_WORK=1 POLYGEIST_BUILD_WORK_DIR="$OUT/nonresident-work" \
  "$HERE/polygeist_build.sh" --target=jetson \
  --function=kernel_llama2_extended_forward \
  --harness="$BENCH/llama_extended_timing_harness.c" "$SOURCE" \
  "${DEFS[@]}" -DWARMUP=5 -DMEASURE=5 -o "$OUT/llama_nonresident"
POLYGEIST_WRAP_KERNEL_PIPELINE=0 POLYGEIST_DISABLE_POINTWISE_MATCHING=0 \
POLYGEIST_KEEP_DEPENDENCIES=1 \
POLYGEIST_CGEIST_FLAGS="--no-inline --memref-fullrank" POLYGEIST_SKIP_LINK=1 \
POLYGEIST_KEEP_WORK=1 POLYGEIST_BUILD_WORK_DIR="$OUT/session-work" \
  "$HERE/polygeist_build.sh" --target=jetson \
  --function=kernel_llama2_extended_forward_session \
  --harness="$BENCH/llama_extended_session_timing_harness.c" "$SOURCE" \
  "${DEFS[@]}" -DSESSION_REPETITIONS=10 -DPOLYGEIST_BENCHMARK_ITERATIONS \
  -o "$OUT/session-objects"
"$CC" -O2 -I"$REPO_ROOT/runtime" -I"$CUDA/include" \
  -c "$BENCH/llama_iteration_timing.c" -o "$OUT/iteration-timing.o"
POLYGEIST_BUILD_WORK_DIR="$OUT/resident-work" \
POLYGEIST_GPU_RESIDUAL_FUNCTION=kernel_llama2_extended_forward \
POLYGEIST_GPU_DATA_RESIDENCY_FUNCTION=kernel_llama2_extended_forward_session \
POLYGEIST_GPU_DATA_RESIDENCY_ASSUME_NO_ALIAS=1 \
POLYGEIST_ENTRY_FUNCTION=kernel_llama2_extended_forward_session \
POLYGEIST_INLINE_GPU_ENTRY_CALLS=1 \
  "$HERE/build_jetson.sh" "$OUT/session-work/abi.mlir" "$OUT/llama_resident" \
  "$OUT/session-work/wrapper.o" "$OUT/session-work/harness.o" \
  "$OUT/session-work/spec_runtime.o" "$OUT/iteration-timing.o"
"$CXX" -O3 -std=c++17 "${DEFS[@]}" "$HERE/llama_extended_ggml_bench.cpp" \
  -I"$REPO_ROOT/third_party/whisper.cpp/ggml/include" \
  -L"$GGML_BUILD/ggml/src" -L"$GGML_BUILD/ggml/src/ggml-cuda" \
  -L"$CUDA/lib" -L"$CUDA/lib/stubs" -Wl,--no-as-needed \
  -lggml -lggml-base -lggml-cpu -lggml-cuda -lcublas -lcudart -lcuda \
  '-Wl,-rpath,$ORIGIN' -Wl,-rpath-link,"$CUDA/lib" -lpthread -ldl -lm \
  -o "$OUT/llama_ggml"
