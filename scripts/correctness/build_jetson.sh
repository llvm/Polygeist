#!/bin/bash
# build_jetson.sh — CROSS-COMPILE a kernel-matched MLIR program on this
# x86_64 dev VM into an aarch64 ELF that runs on a Jetson Orin.
#
# The Jetson does NOT need Polygeist, MLIR, or nvcc — only the CUDA runtime
# libraries that JetPack already installs at /usr/local/cuda/lib64.
#
# See runtime/CROSS_COMPILE.md for the toolchain inventory + why SBSA libs
# work on L4T at runtime.
#
# Usage:
#   ./build_jetson.sh <abi.mlir> <out_exe> [<harness.c|harness.o> ...]
#
# Where <abi.mlir> is the post-Phase-2 IR (already has func.call to
# polygeist_cublas_*, no kernel.launch). Optional harness .c / .o files
# get linked in alongside — pass the C wrapper / main / polybench glue
# here. .c files are compiled with $HARNESS_CFLAGS (default -O3); .o
# files are linked as-is (useful when harness needs project-specific
# preprocessor defines like -DPOLYBENCH_USE_C99_PROTO that you've already
# baked into a pre-built .o on the host).
#
# Output: aarch64-linux-gnu ELF with DT_NEEDED on libcublas.so.12 +
# libcudart.so.12, RUNPATH=/usr/local/cuda/lib64.
#
# scp the binary to the Jetson and run:
#   ./<out_exe>
# Or profile with nsys (on the Jetson):
#   nsys profile -o trace ./<out_exe>

set -euo pipefail
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <abi.mlir> <out_exe> [<harness.c> ...]" >&2
  exit 1
fi

INPUT=$1
OUT_EXE=$2
shift 2
HARNESS=("$@")
OUT_DIR=$(dirname "$OUT_EXE")
mkdir -p "$OUT_DIR"

# Optional preprocessor / opt flags forwarded to .c harness compilation only.
# Pre-built .o files are linked as-is. Use this for polybench-style defines.
HARNESS_CFLAGS="${HARNESS_CFLAGS:--O3}"

# ─── Cross toolchain (host: x86_64; target: aarch64 + Jetson CUDA) ─────────
# Override these via env vars if the cross-toolkit lives elsewhere.
CUDA_CROSS_VER=${CUDA_CROSS_VER:-12.6}
CUDA=${CUDA:-/usr/local/cuda-${CUDA_CROSS_VER}/targets/sbsa-linux}
AARCH64_CC=${AARCH64_CC:-aarch64-linux-gnu-gcc}
AARCH64_READELF=${AARCH64_READELF:-aarch64-linux-gnu-readelf}
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
CLANG=$REPO_ROOT/llvm-project/build/bin/clang
RT=$REPO_ROOT/runtime

# Sanity checks
for tool in "$AARCH64_CC" "$AARCH64_READELF"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: $tool not on PATH. Install gcc-aarch64-linux-gnu." >&2
    echo "       See runtime/CROSS_COMPILE.md." >&2
    exit 1
  fi
done
if [ ! -d "$CUDA/include" ] || [ ! -d "$CUDA/lib" ]; then
  echo "ERROR: CUDA cross-toolkit not found at $CUDA" >&2
  echo "       Install cuda-cudart-cross-sbsa-* + libcublas-cross-sbsa-* +" >&2
  echo "       cuda-nvcc-cross-sbsa-* (for crt/ headers)." >&2
  echo "       See runtime/CROSS_COMPILE.md." >&2
  exit 1
fi
if [ ! -s "$INPUT" ]; then
  echo "ERROR: input MLIR '$INPUT' is missing or empty" >&2
  exit 1
fi

# Reject obviously-not-ABI-lowered input. Saves an obscure later failure.
if grep -q '= kernel\.launch ' "$INPUT"; then
  echo "ERROR: $INPUT still has kernel.launch ops — run" >&2
  echo "       polygeist-opt --lower-kernel-launch-to-cublas first." >&2
  exit 1
fi

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

echo "  [1/6] copy + canonicalise input MLIR"
# Mark to_tensor results as `restrict` so one-shot-bufferize keeps the
# in-place semantics (same trick gemm_kernel_e2e.sh uses).
sed 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
    "$INPUT" > $WORK/abi.mlir

echo "  [2/6] one-shot-bufferize + lower to LLVM dialect (host-side, on this VM)"
$MLIR_OPT --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $WORK/abi.mlir -o $WORK/llvm.mlir

echo "  [3/6] translate to LLVM IR, then retarget x86 → aarch64"
$MLIR_TRANSLATE --mlir-to-llvmir $WORK/llvm.mlir -o $WORK/kernel.ll
# Rewrite the embedded target triple so clang doesn't think this is x86
# when we feed it through with --target=aarch64. Drop the datalayout
# line entirely; clang will re-derive an aarch64 layout.
sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|' \
    $WORK/kernel.ll
sed -i '/^target datalayout/d' $WORK/kernel.ll
# `kernel_gemm` is what the polybench harness will call — rename so the
# harness's own `kernel_gemm` (the C ref) doesn't collide.
sed -i 's/@kernel_gemm\b/@kernel_gemm_impl/g' $WORK/kernel.ll

echo "  [4/6] cross-compile .ll → aarch64 .o via Polygeist clang"
$CLANG --target=aarch64-linux-gnu --gcc-toolchain=/usr \
       -O3 -c $WORK/kernel.ll -o $WORK/kernel.o

echo "  [5/6] cross-compile runtime shim + any harness .c files"
# The shim now includes cuDNN for conv2d; cuDNN headers live in the
# aarch64 cross-dev location, separate from CUDA's include path.
CUDNN_INC=${CUDNN_INC:-/usr/include/aarch64-linux-gnu}
CUDNN_LIB=${CUDNN_LIB:-/usr/lib/aarch64-linux-gnu}
$AARCH64_CC -O3 -I$CUDA/include -I$CUDNN_INC -c \
            $RT/polygeist_cublas_rt_cuda.c -o $WORK/rt.o
HARNESS_OBJS=()
for item in "${HARNESS[@]}"; do
  case "$item" in
    *.c)
      obj=$WORK/$(basename "$item" .c).o
      echo "       harness (compile): $item → $(basename $obj)"
      $AARCH64_CC $HARNESS_CFLAGS -c "$item" -o "$obj"
      HARNESS_OBJS+=("$obj")
      ;;
    *.o)
      echo "       harness (pre-built): $item"
      HARNESS_OBJS+=("$item")
      ;;
    *)
      echo "ERROR: harness arg must be .c or .o file: $item" >&2
      exit 1
      ;;
  esac
done

echo "  [6/6] link against aarch64 cuBLAS + cudart stubs"
# Stub libs live in $CUDA/lib (for libcudart) and $CUDA/lib/stubs (for
# libcublas). Both are aarch64 ELF; the actual .so files resolve against
# JetPack's installed CUDA at runtime via RUNPATH.
$AARCH64_CC -O2 \
    $WORK/kernel.o $WORK/rt.o "${HARNESS_OBJS[@]}" \
    -L$CUDA/lib -L$CUDA/lib/stubs -L$CUDNN_LIB \
    -lcudnn -lcublasLt -lcublas -lcudart -lm -lpthread -ldl \
    -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
    -o "$OUT_EXE"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Cross-build complete:"
file "$OUT_EXE"
echo ""
echo "DT_NEEDED (must show libcublas.so.12 + libcudart.so.12):"
$AARCH64_READELF -d "$OUT_EXE" | grep -E 'NEEDED|RUNPATH'
echo ""
echo "Binary size: $(stat -c '%s bytes' "$OUT_EXE")"
echo ""
echo "Ship to Jetson with:"
echo "  scp '$OUT_EXE' nvidia@<jetson>:/tmp/"
echo "  ssh nvidia@<jetson> 'chmod +x /tmp/$(basename "$OUT_EXE") && /tmp/$(basename "$OUT_EXE")'"
echo ""
echo "Or profile on Jetson with nsys:"
echo "  ssh nvidia@<jetson> 'nsys profile -o /tmp/trace /tmp/$(basename "$OUT_EXE")'"
echo "═══════════════════════════════════════════════════════════════════════"
