#!/bin/bash
# Cross-build upstream cuSten plus Polygeist's ABI-only adapter for Jetson.
# The resulting shared library is copied to the board with the executable;
# no compilation happens on the board.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUSTEN_ROOT="${POLYGEIST_CUSTEN_ROOT:-$REPO_ROOT/third_party/cuSten}"
CUDA_ROOT="${POLYGEIST_CUDA_CROSS_ROOT:-/usr/local/cuda-12.6}"
NVCC="${POLYGEIST_NVCC:-$CUDA_ROOT/bin/nvcc}"
CROSS_CXX="${POLYGEIST_AARCH64_CXX:-aarch64-linux-gnu-g++}"
GPU_ARCH="${POLYGEIST_GPU_ARCH:-sm_87}"
OUT="${1:-$REPO_ROOT/build/custen-jetson/libpolygeist_custen.so}"

[ -x "$NVCC" ] || {
  echo "ERROR: cross-capable nvcc not found at $NVCC" >&2
  echo "Install the CUDA cross compiler locally; do not build cuSten on Jetson." >&2
  exit 1
}
command -v "$CROSS_CXX" >/dev/null || {
  echo "ERROR: $CROSS_CXX not found" >&2
  exit 1
}
[ -f "$CUSTEN_ROOT/cuSten/cuSten.h" ] || {
  echo "ERROR: initialize third_party/cuSten (git submodule update --init)" >&2
  exit 1
}

mkdir -p "$(dirname "$OUT")"
mapfile -t CUSTEN_SOURCES < <(
  find "$CUSTEN_ROOT/cuSten/src" -type f -name '*.cu' | sort
)

"$NVCC" -O3 -std=c++11 -arch="$GPU_ARCH" -ccbin "$CROSS_CXX" \
  -Xcompiler=-fPIC -shared \
  -DcudaMemPrefetchAsync=polygeist_custen_prefetch \
  -I"$CUSTEN_ROOT/cuSten" \
  -I"$CUSTEN_ROOT/cuSten/src/struct" \
  -I"$CUSTEN_ROOT/cuSten/src/kernels" \
  -I"$CUSTEN_ROOT/cuSten/src/util" \
  -I"$REPO_ROOT/runtime" \
  "${CUSTEN_SOURCES[@]}" "$REPO_ROOT/runtime/polygeist_custen_adapter.cu" \
  -o "$OUT"

file "$OUT"
