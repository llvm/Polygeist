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

# Optional cuTensorNet cross package. NVIDIA's Python wheels use separate
# cuquantum/ and cutensor/ roots, so callers may either provide the three
# explicit directories or a common extraction root with that layout.
CUTENSORNET_ROOT="${POLYGEIST_CUTENSORNET_ROOT:-}"
CUTENSORNET_INC="${POLYGEIST_CUTENSORNET_INCLUDE:-}"
CUTENSORNET_LIB="${POLYGEIST_CUTENSORNET_LIBDIR:-}"
CUTENSOR_LIB="${POLYGEIST_CUTENSOR_LIBDIR:-}"
if [ -n "$CUTENSORNET_ROOT" ]; then
  CUTENSORNET_INC="${CUTENSORNET_INC:-$CUTENSORNET_ROOT/cuquantum/include}"
  CUTENSORNET_LIB="${CUTENSORNET_LIB:-$CUTENSORNET_ROOT/cuquantum/lib}"
  CUTENSOR_LIB="${CUTENSOR_LIB:-$CUTENSORNET_ROOT/cutensor/lib}"
fi
CUTENSORNET_ENABLED=0
if [ -n "$CUTENSORNET_INC" ] || [ -n "$CUTENSORNET_LIB" ] || \
   [ -n "$CUTENSOR_LIB" ]; then
  for required in "$CUTENSORNET_INC/cutensornet.h" \
                  "$CUTENSORNET_LIB/libcutensornet.so.2" \
                  "$CUTENSOR_LIB/libcutensor.so.2"; do
    [ -f "$required" ] || {
      echo "ERROR: cuTensorNet cross-build input missing: $required" >&2
      exit 1
    }
  done
  CUTENSORNET_ENABLED=1
fi

# ─── Cross toolchain (host: x86_64; target: aarch64 + Jetson CUDA) ─────────
# Override these via env vars if the cross-toolkit lives elsewhere.
CUDA_CROSS_VER=${CUDA_CROSS_VER:-12.6}
CUDA=${CUDA:-/usr/local/cuda-${CUDA_CROSS_VER}/targets/sbsa-linux}
AARCH64_CC=${AARCH64_CC:-aarch64-linux-gnu-gcc}
AARCH64_READELF=${AARCH64_READELF:-aarch64-linux-gnu-readelf}
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
POLYGEIST_OPT=$REPO_ROOT/build/bin/polygeist-opt
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

if [ -n "${POLYGEIST_BUILD_WORK_DIR:-}" ]; then
  WORK=$POLYGEIST_BUILD_WORK_DIR
  mkdir -p "$WORK"
  rm -f "$WORK"/*.mlir "$WORK"/*.ll "$WORK"/*.o
else
  WORK=$(mktemp -d)
  trap "rm -rf $WORK" EXIT
fi

echo "  [1/6] lower Polygeist views + canonicalise input MLIR"
# Tensor-form matcher results can retain polygeist.submap/submapInverse around
# an already ABI-lowered runtime call. Lower those repository-specific ops
# before handing the module to upstream mlir-opt, which does not register the
# Polygeist dialect.
$POLYGEIST_OPT --lower-polygeist-submap "$INPUT" -o $WORK/no_submap_raw.mlir
# Recover static allocation types hidden by tensor casts/dim queries. Besides
# reducing descriptor traffic, this lets the optional persistent-workspace
# planner recognize ABI compatibility snapshots without application shapes.
$MLIR_OPT --resolve-shaped-type-result-dims --canonicalize \
  $WORK/no_submap_raw.mlir -o $WORK/no_submap.mlir
# Graph-captured applications may request stable storage for static scratch
# created while lowering Polygeist views. This is deliberately opt-in because
# module-global workspaces make the selected function non-reentrant.
ABI_INPUT=$WORK/no_submap.mlir
if [ -n "${POLYGEIST_PERSISTENT_WORKSPACE_FUNCTION:-}" ]; then
  $POLYGEIST_OPT \
    "--plan-persistent-gpu-workspace=function=${POLYGEIST_PERSISTENT_WORKSPACE_FUNCTION}" \
    $WORK/no_submap.mlir -o $WORK/persistent_workspace.mlir
  ABI_INPUT=$WORK/persistent_workspace.mlir
fi
# Mark to_tensor results as `restrict` so one-shot-bufferize keeps the
# in-place semantics (same trick gemm_kernel_e2e.sh uses).
sed 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
    $ABI_INPUT > $WORK/abi.mlir

echo "  [2/6] one-shot-bufferize + lower to LLVM dialect (host-side, on this VM)"
if [ -n "${POLYGEIST_GPU_RESIDUAL_FUNCTION:-}" ]; then
  # Preserve bufferized Linalg long enough to outline every unmatched
  # parallel/reduction stage and every compatibility copy as generated GPU
  # code. The preparation pass runs on both sides of the standard MLIR
  # outlining pipeline: first to expose copies/injective write-backs, then to
  # register mapped bases and mark launches capture-safe.
  GPU_FN=$POLYGEIST_GPU_RESIDUAL_FUNCTION
  GPU_ARCH=${POLYGEIST_GPU_ARCH:-sm_87}
  $MLIR_OPT --empty-tensor-to-alloc-tensor \
    --one-shot-bufferize=bufferize-function-boundaries \
    --canonicalize --promote-buffers-to-stack \
    $WORK/abi.mlir -o $WORK/bufferized.mlir
  $POLYGEIST_OPT \
    "--prepare-gpu-residual-pipeline=function=${GPU_FN}" \
    $WORK/bufferized.mlir -o $WORK/gpu_prepared.mlir
  $MLIR_OPT --convert-linalg-to-parallel-loops \
    --gpu-map-parallel-loops --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    $WORK/gpu_prepared.mlir -o $WORK/gpu_outlined.mlir
  $POLYGEIST_OPT --merge-gpu-modules \
    $WORK/gpu_outlined.mlir -o $WORK/gpu_merged.mlir
  $POLYGEIST_OPT \
    "--prepare-gpu-residual-pipeline=function=${GPU_FN}" \
    $WORK/gpu_merged.mlir -o $WORK/gpu_registered.mlir
  $POLYGEIST_OPT \
    '--wrap-kernel-launch-pipeline=cuda-graphs=true capture-host-mapped-cutensornet=true maximal-device-sequence=true' \
    $WORK/gpu_registered.mlir -o $WORK/gpu_graphed.mlir
  # Current mlir-opt rejects combining a nested --pass-pipeline with
  # individual top-level pass flags. Attach the NVPTX target in its own
  # invocation, then serialize modules and lower the host side.
  $MLIR_OPT \
    --pass-pipeline="builtin.module(gpu.module(affine-expand-index-ops,lower-affine,convert-scf-to-cf,convert-gpu-to-nvvm,convert-arith-to-llvm,convert-index-to-llvm),convert-cf-to-llvm,gpu.module(canonicalize,cse),nvvm-attach-target{chip=${GPU_ARCH} O=3})" \
    $WORK/gpu_graphed.mlir -o $WORK/gpu_targeted.mlir
  $MLIR_OPT \
    --gpu-to-llvm --gpu-module-to-binary=format=isa \
    --expand-strided-metadata --lower-affine --convert-scf-to-cf \
    --convert-arith-to-llvm --convert-index-to-llvm \
    --finalize-memref-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    $WORK/gpu_targeted.mlir -o $WORK/llvm.mlir
else
  $MLIR_OPT --empty-tensor-to-alloc-tensor \
    --one-shot-bufferize=bufferize-function-boundaries \
    --canonicalize --promote-buffers-to-stack \
    --convert-linalg-to-loops --expand-strided-metadata --lower-affine \
    --convert-scf-to-cf --convert-arith-to-llvm --convert-index-to-llvm \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts \
    $WORK/abi.mlir -o $WORK/llvm.mlir
fi

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
# Application harnesses expose a flat-pointer C entry point while the raised
# MLIR function uses the expanded memref ABI. Keep the same convention as the
# GEMM harness for any explicitly selected residual-GPU application.
if [ -n "${POLYGEIST_GPU_RESIDUAL_FUNCTION:-}" ]; then
  GPU_FN=$POLYGEIST_GPU_RESIDUAL_FUNCTION
  sed -i "s/@${GPU_FN}\\b/@${GPU_FN}_impl/g" $WORK/kernel.ll
fi

echo "  [4/6] cross-compile .ll → aarch64 .o via Polygeist clang"
$CLANG --target=aarch64-linux-gnu --gcc-toolchain=/usr \
       -O3 -c $WORK/kernel.ll -o $WORK/kernel.o

echo "  [5/6] cross-compile runtime shim + any harness .c files"
# The shim now includes cuDNN for conv2d; cuDNN headers live in the
# aarch64 cross-dev location, separate from CUDA's include path.
CUDNN_INC=${CUDNN_INC:-/usr/include/aarch64-linux-gnu}
CUDNN_LIB=${CUDNN_LIB:-/usr/lib/aarch64-linux-gnu}
RT_EXTRA_CFLAGS=()
RT_EXTRA_LIBS=()
RT_LINK_FLAGS=()
ACCEL_LIBS=(-lcudnn -lcublasLt -lcublas -lcufft -lcusparse -lcusolver)
if [ "${POLYGEIST_MINIMAL_RUNTIME:-0}" != "0" ]; then
  # Put every runtime entry point in its own ELF section and discard entries
  # unreachable from this executable. This lets a cuTensorNet-only binary run
  # on minimal JetPack images that do not install cuDNN/cuFFT.
  RT_EXTRA_CFLAGS+=("-ffunction-sections" "-fdata-sections")
  RT_LINK_FLAGS+=("-Wl,--gc-sections")
  ACCEL_LIBS=(-lcublasLt -lcublas -lcusolver)
  echo "       minimal runtime: dead-strip unused library shims"
fi
if [ "$CUTENSORNET_ENABLED" -eq 1 ]; then
  RT_EXTRA_CFLAGS+=("-DPOLYGEIST_ENABLE_CUTENSORNET" "-I$CUTENSORNET_INC")
  RT_EXTRA_LIBS+=("-L$CUTENSORNET_LIB" "-L$CUTENSOR_LIB"
                 "-l:libcutensornet.so.2" "-l:libcutensor.so.2")
  echo "       cuTensorNet: $CUTENSORNET_LIB"
  echo "       cuTENSOR:    $CUTENSOR_LIB"
fi
# Standalone cuTENSOR (for cutensorUnary etc.) — enable when a cutensor root
# (include/ + lib/libcutensor.so.2) is provided, independent of cuTensorNet.
if [ -z "${POLYGEIST_CUTENSOR_ROOT:-}" ] && [ "$CUTENSORNET_ENABLED" -ne 1 ]; then
  :
elif [ -n "${POLYGEIST_CUTENSOR_ROOT:-}" ]; then
  RT_EXTRA_CFLAGS+=("-DPOLYGEIST_ENABLE_CUTENSOR" "-I$POLYGEIST_CUTENSOR_ROOT/include")
  RT_EXTRA_LIBS+=("-L$POLYGEIST_CUTENSOR_ROOT/lib" "-l:libcutensor.so.2")
  echo "       cuTENSOR (standalone): $POLYGEIST_CUTENSOR_ROOT"
fi
$AARCH64_CC -O3 -I$CUDA/include -I$CUDNN_INC "${RT_EXTRA_CFLAGS[@]}" \
            -c $RT/polygeist_cublas_rt_cuda.c -o $WORK/rt.o
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
    "${RT_EXTRA_LIBS[@]}" \
    "${ACCEL_LIBS[@]}" -lcudart \
    "${RT_LINK_FLAGS[@]}" \
    -lm -lpthread -ldl \
    '-Wl,-rpath,$ORIGIN:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu' \
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
