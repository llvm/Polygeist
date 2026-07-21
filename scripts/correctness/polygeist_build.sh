#!/bin/bash
# polygeist_build.sh — generic driver: take a C source file containing a
# kernel function and produce a binary where the kernel is matched to an
# optimized library implementation (cuDNN / cuBLAS) and the rest of the
# file (main, init, print, etc.) is compiled normally.
#
# Usage:
#   polygeist_build.sh [--target=host|jetson] [--function=NAME] [-o OUT]
#                      [--harness=HARNESS.c] [--no-debuf]
#                      <kernel.c> [gcc-passthrough-flags...]
#
# Defaults:
#   --target=host       Produce a binary for the local machine. On an x86
#                       dev VM with no CUDA, links the CPU-stub runtime so
#                       the binary still runs (CPU-only, for correctness).
#                       On a Jetson (aarch64 + JetPack CUDA), links cuDNN/
#                       cuBLAS and the binary runs on the GPU.
#   --target=jetson     Cross-compile from this x86 VM to aarch64 + bundle
#                       the cross-CUDA libs. The resulting binary is an
#                       aarch64 ELF you can scp to a Jetson and run there.
#                       Deployment (scp / ssh / execute) is out of scope
#                       for this driver — that's a separate, environment-
#                       specific concern.
#   --function=auto     Auto-detect the kernel function via #pragma scop
#                       (PolyBench convention) or a leading 'kernel_' prefix.
#                       Override with --function=NAME for non-conventional
#                       source.
#   -o OUT              Defaults to the .c basename without extension.
#   --no-debuf          Match the memref linalg form directly instead of
#                       running --linalg-debufferize before the matcher.
#                       Useful for memref-only compositions such as the
#                       llama2.c RMSNorm/softmax patterns.
#
# Optional environment:
#   POLYGEIST_CPU_BLAS=1
#                       Host target only. Compile the CPU runtime shim with
#                       CBLAS calls for BLAS-like symbols and link OpenBLAS by
#                       default. Override with POLYGEIST_CPU_BLAS_CFLAGS and
#                       POLYGEIST_CPU_BLAS_LIBS for MKL/BLIS/ArmPL/NVPL.
#   POLYGEIST_CUTENSORNET_ROOT=/path/to/cuquantum
#                       Jetson target only. The root must contain
#                       include/cutensornet.h and lib/libcutensornet.so for
#                       aarch64. Enables the cuTensorNet tensor-product shim.
#
# Any unrecognized flags are passed through to all the gcc/clang invocations
# that compile non-MLIR pieces of the build (harness, polybench utility code,
# runtime shim). This is how PolyBench-style preprocessor defines like
# -DMINI_DATASET / -DDATA_TYPE_IS_DOUBLE / -DPOLYBENCH_DUMP_ARRAYS get
# propagated — they're just gcc flags from the driver's perspective.
#
# Examples:
#   polygeist_build.sh gemm.c -DMINI_DATASET -I /path/polybench/utilities
#   polygeist_build.sh --target=jetson gemm.c -DLARGE_DATASET -o gemm_jetson
#   polygeist_build.sh --function=kernel_conv2d conv2d.c

set -euo pipefail
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

# ─── Tooling ────────────────────────────────────────────────────────────
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
CLANG=$REPO_ROOT/llvm-project/build/bin/clang
PYTHON=$PYTHON
SCRIPTS=$REPO_ROOT/scripts/correctness
RT=$REPO_ROOT/runtime
KERNEL_LIB=$REPO_ROOT/generic_solver/kernel_library_phase2.mlir

# Cross toolchain (used only when --target=jetson).
CUDA_CROSS=/usr/local/cuda-12.6/targets/sbsa-linux
CUDNN_CROSS_INC=/usr/include/aarch64-linux-gnu
CUDNN_CROSS_LIB=/usr/lib/aarch64-linux-gnu
AARCH64_CC=aarch64-linux-gnu-gcc

# ─── Parse args ─────────────────────────────────────────────────────────
TARGET=host
FUNCTION=
OUT=
INPUT=
HARNESS_INPUT=
DEBUFFERIZE=1
GCC_PASSTHROUGH=()
RT_CFLAGS=()

usage() {
  sed -n '3,40p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --target=*)    TARGET="${1#--target=}"; shift ;;
    --function=*)  FUNCTION="${1#--function=}"; shift ;;
    --harness=*)   HARNESS_INPUT="${1#--harness=}"; shift ;;
    --no-debuf|--no-linalg-debufferize) DEBUFFERIZE=0; shift ;;
    -o)            OUT="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *.c)
      if [ -z "$INPUT" ]; then INPUT="$1"
      else GCC_PASSTHROUGH+=("$1"); fi
      shift ;;
    *)             GCC_PASSTHROUGH+=("$1"); shift ;;
  esac
done

[ -z "$INPUT" ] && { echo "ERROR: no .c input file provided" >&2; usage 1; }
[ -f "$INPUT" ] || { echo "ERROR: input file $INPUT not found" >&2; exit 1; }
[ -n "$HARNESS_INPUT" ] || HARNESS_INPUT="$INPUT"
[ -f "$HARNESS_INPUT" ] || { echo "ERROR: harness file $HARNESS_INPUT not found" >&2; exit 1; }
case "$TARGET" in host|jetson) ;; *)
  echo "ERROR: --target must be 'host' or 'jetson' (got '$TARGET')" >&2; exit 1 ;;
esac
[ -z "$OUT" ] && OUT="$(basename "$INPUT" .c)"

# ─── Auto-detect the kernel function name ───────────────────────────────
if [ -z "$FUNCTION" ]; then
  # Strategy 1: find the function immediately preceding '#pragma scop'
  # (PolyBench convention — the scop marker sits in the kernel function body).
  FUNCTION=$(awk '
    /^void\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(/ {
      match($0, /^void\s+([a-zA-Z_][a-zA-Z0-9_]*)/, a); last_fn = a[1]
    }
    /#pragma\s+scop/ { print last_fn; exit }
  ' "$INPUT")
  # Strategy 2: first function whose name starts with kernel_
  if [ -z "$FUNCTION" ]; then
    FUNCTION=$(grep -oE '^\s*(static\s+)?void\s+kernel_[a-zA-Z0-9_]+' "$INPUT" \
               | head -1 | awk '{print $NF}')
  fi
  if [ -z "$FUNCTION" ]; then
    echo "ERROR: couldn't auto-detect kernel function in $INPUT." >&2
    echo "       Use --function=NAME to specify it explicitly." >&2
    exit 1
  fi
fi

WORK=$(mktemp -d)
if [ "${POLYGEIST_KEEP_WORK:-0}" != "0" ]; then
  echo "[polygeist] keeping workdir: $WORK"
else
  trap "rm -rf $WORK" EXIT
fi

echo "[polygeist] input=$INPUT  function=$FUNCTION  target=$TARGET  output=$OUT"
echo "[polygeist] harness=$HARNESS_INPUT"
echo "[polygeist] gcc passthrough: ${GCC_PASSTHROUGH[*]:-(none)}"

# ─── Step 1: cgeist lifts the kernel function to affine MLIR ────────────
echo "  [1/9] cgeist → affine MLIR"
cgeist "$INPUT" --function="$FUNCTION" \
  --resource-dir=/usr/lib/clang/14 \
  "${GCC_PASSTHROUGH[@]}" \
  --raise-scf-to-affine -fPIC -S \
  -o $WORK/affine.mlir 2>$WORK/cgeist.err || {
    echo "ERROR: cgeist failed; see $WORK/cgeist.err" >&2; cat $WORK/cgeist.err >&2; exit 1; }

# ─── Step 2: raise affine → linalg + debufferize ────────────────────────
if [ "$DEBUFFERIZE" -eq 1 ]; then
  echo "  [2/9] polygeist-opt: raise + lower-submap + debufferize"
  polygeist-opt --select-func=func-name="$FUNCTION" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline \
    --lower-polygeist-submap \
    --linalg-debufferize \
    $WORK/affine.mlir -o $WORK/linalg.mlir 2>$WORK/raise.err || {
      echo "ERROR: raise pass failed; see $WORK/raise.err" >&2; cat $WORK/raise.err >&2; exit 1; }
else
  echo "  [2/9] polygeist-opt: raise + lower-submap (memref linalg)"
  polygeist-opt --select-func=func-name="$FUNCTION" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline \
    --lower-polygeist-submap \
    $WORK/affine.mlir -o $WORK/linalg.mlir 2>$WORK/raise.err || {
      echo "ERROR: raise pass failed; see $WORK/raise.err" >&2; cat $WORK/raise.err >&2; exit 1; }
fi

# ─── Step 3: matcher (linalg.generic → kernel.launch) ───────────────────
echo "  [3/9] matcher: linalg.generic → kernel.launch"
$PYTHON $SCRIPTS/kernel_match_rewrite.py \
  $WORK/linalg.mlir > $WORK/matched.mlir 2>$WORK/match.err
N_LAUNCH=$(grep -c 'kernel\.launch' $WORK/matched.mlir || true)
echo "         matched $N_LAUNCH kernel.launch op(s)"
if [ "${N_LAUNCH:-0}" -eq 0 ]; then
  echo "         no ABI-lowerable matches; continuing with residual Linalg"
fi

# ─── Step 4: inject canonical kernel.defn declarations ──────────────────
# The matched MLIR references @cublasDgemm / @cudnnConvolution2D_9tap / etc.
# but doesn't define them. The kernel.launch op's verifier needs the symbols
# to exist. We pull all the kernel.defn entries from kernel_library_phase2.mlir
# and inject them inside the matched module's attribute block. The lowering
# pass dead-strips unused defns afterwards, so injecting all of them is safe
# regardless of which one(s) the matcher emitted.
echo "  [4/9] inject canonical defns from kernel_library_phase2.mlir"
# Extract the kernel.defn blocks from the library (everything between the
# outer module { ... }), strip the wrapping module line, and inject.
DEFNS=$(sed -n '/^module {$/,/^}$/p' "$KERNEL_LIB" | sed '1d; $d')
awk -v defns="$DEFNS" '
  /^module attributes/ && !done { print; print defns; done=1; next }
  { print }
' $WORK/matched.mlir > $WORK/with_defns.mlir

# ─── Step 5: ABI lowering kernel.launch → func.call to runtime shim ─────
echo "  [5/9] polygeist-opt: lower-kernel-launch-to-cublas (kernel.launch → func.call)"
ABI_PASSES=(--lower-kernel-launch-to-cublas)
if [ "${POLYGEIST_WRAP_KERNEL_PIPELINE:-0}" != "0" ]; then
  ABI_PASSES+=(--wrap-kernel-launch-pipeline)
fi
polygeist-opt "${ABI_PASSES[@]}" \
  $WORK/with_defns.mlir -o $WORK/abi.mlir 2>$WORK/abi.err || {
    echo "ERROR: ABI lowering failed; see $WORK/abi.err" >&2; cat $WORK/abi.err >&2; exit 1; }
N_CALL=$(grep -cE 'call @polygeist_' $WORK/abi.mlir || true)
echo "         emitted $N_CALL func.call to runtime shim"

# ─── Step 6: lower to LLVM dialect + translate to LLVM IR ───────────────
echo "  [6/9] mlir-opt → LLVM dialect → llvm-translate → kernel.ll"
# ABI lowering can leave pure polygeist.submap/submapInverse view ops around,
# especially when a matched launch consumed one view but the neighboring CPU
# residual linalg still uses another. Clean those up with polygeist-opt before
# handing the IR to upstream mlir-opt, which does not load the Polygeist dialect.
polygeist-opt --canonicalize --cse --lower-polygeist-submap --canonicalize --cse \
  $WORK/abi.mlir -o $WORK/abi_canon.mlir 2>>$WORK/abi.err || {
    echo "ERROR: polygeist submap cleanup failed; see $WORK/abi.err" >&2
    cat $WORK/abi.err >&2
    exit 1
  }
# Mark to_tensor results restrict so one-shot-bufferize keeps in-place semantics.
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
  $WORK/abi_canon.mlir
$MLIR_OPT --convert-math-to-llvm \
  --empty-tensor-to-alloc-tensor \
  --lower-affine \
  --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $WORK/abi_canon.mlir -o $WORK/llvm.mlir 2>$WORK/mlir.err || {
    echo "ERROR: mlir-opt lowering failed; see $WORK/mlir.err" >&2; cat $WORK/mlir.err >&2; exit 1; }
$MLIR_TRANSLATE --mlir-to-llvmir $WORK/llvm.mlir -o $WORK/kernel.ll

# Rename the lifted symbol to <name>_impl so the harness's own C definition
# of the same function name doesn't collide. The auto-generated wrapper
# provides the public <name> entry that calls _impl with packed memrefs.
sed -i "s/@${FUNCTION}\b/@${FUNCTION}_impl/g" $WORK/kernel.ll

# Retarget the LLVM IR if we're cross-compiling. clang's --target flag will
# also do most of this, but stripping the embedded x86 datalayout avoids
# warnings and lets clang re-derive an aarch64 layout from --target.
if [ "$TARGET" = "jetson" ]; then
  sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|' $WORK/kernel.ll
  sed -i '/^target datalayout/d' $WORK/kernel.ll
fi

# ─── Step 7: generate the ABI wrapper for the kernel ────────────────────
echo "  [7/9] gen_wrapper.py: ABI bridge for $FUNCTION"
$PYTHON $SCRIPTS/gen_wrapper.py "$INPUT" "$FUNCTION" > $WORK/wrapper.c

# ─── Step 8: per-target compile + harness prep ──────────────────────────
echo "  [8/9] compile kernel.ll + wrapper + harness + runtime shim (target=$TARGET)"
if [ "$TARGET" = "host" ]; then
  CC=$CLANG
  CLANG_TARGET_ARGS=""
  RT_SRC=$RT/polygeist_cublas_rt_cpu.c
  RT_LIBS="-lm -lpthread"
  if [ "${POLYGEIST_CPU_BLAS:-0}" != "0" ]; then
    RT_CFLAGS+=("-DPOLYGEIST_CPU_USE_CBLAS")
    if [ -n "${POLYGEIST_CPU_BLAS_CFLAGS:-}" ]; then
      read -r -a _CPU_BLAS_CFLAGS <<< "$POLYGEIST_CPU_BLAS_CFLAGS"
      RT_CFLAGS+=("${_CPU_BLAS_CFLAGS[@]}")
    fi
    RT_LIBS="${POLYGEIST_CPU_BLAS_LIBS:--lopenblas} $RT_LIBS"
    echo "         + optimized CPU CBLAS runtime enabled"
  fi
else
  # aarch64-linux-gnu-gcc is already configured for aarch64 — no --target arg.
  # Clang (used for kernel.ll → kernel.o only) does need --target=aarch64-linux-gnu.
  CC=$AARCH64_CC
  CLANG_TARGET_ARGS="--target=aarch64-linux-gnu --gcc-toolchain=/usr"
  RT_SRC=$RT/polygeist_cublas_rt_cuda.c
  RT_LIBS="-L$CUDA_CROSS/lib -L$CUDA_CROSS/lib/stubs -L$CUDNN_CROSS_LIB \
           -lcudnn -lcublasLt -lcublas -lcufft -lcusparse -lcusolver \
           -lcudart -lm -lpthread -ldl \
           -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"
  if [ -n "${POLYGEIST_CUTENSORNET_ROOT:-}" ]; then
    CUTENSORNET_ROOT=$POLYGEIST_CUTENSORNET_ROOT
    [ -f "$CUTENSORNET_ROOT/include/cutensornet.h" ] || {
      echo "ERROR: $CUTENSORNET_ROOT/include/cutensornet.h not found" >&2
      exit 1
    }
    RT_CFLAGS+=("-DPOLYGEIST_ENABLE_CUTENSORNET"
               "-I$CUTENSORNET_ROOT/include")
    RT_LIBS="-L$CUTENSORNET_ROOT/lib -lcutensornet -lcutensor $RT_LIBS"
    echo "         + cuTensorNet runtime from $CUTENSORNET_ROOT"
  fi
fi

# Kernel (lifted) — use Polygeist clang for both host and cross.
$CLANG $CLANG_TARGET_ARGS -O3 -c $WORK/kernel.ll -o $WORK/kernel.o

# Wrapper (ABI bridge generated by gen_wrapper.py).
$CC -O2 "${GCC_PASSTHROUGH[@]}" -c $WORK/wrapper.c -o $WORK/wrapper.o

# Harness compiled normally. If it is the original source and defines the
# selected kernel, weaken that symbol so the lifted+matched wrapper wins.
# Separate harness files only declare/call the kernel, so no weakening is
# needed and the compiler cannot inline the original body into main.
$CC -O0 -fno-inline -fno-inline-functions "${GCC_PASSTHROUGH[@]}" \
  -c "$HARNESS_INPUT" -o $WORK/harness_full.o
NM_TOOL=nm
if [ "$TARGET" = "jetson" ] && command -v aarch64-linux-gnu-nm >/dev/null 2>&1; then
  NM_TOOL=aarch64-linux-gnu-nm
fi
if $NM_TOOL $WORK/harness_full.o | awk '{print $3}' | grep -qx "$FUNCTION"; then
  if [ "$TARGET" = "host" ]; then
    objcopy --weaken-symbol="$FUNCTION" $WORK/harness_full.o $WORK/harness.o
  else
    aarch64-linux-gnu-objcopy --weaken-symbol="$FUNCTION" \
      $WORK/harness_full.o $WORK/harness.o
  fi
else
  cp $WORK/harness_full.o $WORK/harness.o
fi

# Runtime shim. For jetson target we also need cuda + cudnn headers.
if [ "$TARGET" = "host" ]; then
  $CC -O2 "${RT_CFLAGS[@]}" -c $RT_SRC -o $WORK/rt.o
  $CC -O2 -c $RT/polygeist_mlir_runner_utils.c -o $WORK/mlir_runner_utils.o
else
  $CC -O2 "${RT_CFLAGS[@]}" -I$CUDA_CROSS/include -I$CUDNN_CROSS_INC \
    -c $RT_SRC -o $WORK/rt.o
  $CC -O2 -c $RT/polygeist_mlir_runner_utils.c -o $WORK/mlir_runner_utils.o
fi

# Polybench utility .c — only if the harness uses POLYBENCH macros and the
# user provided -I to its include path. Detect via 'polybench.h' include.
POLYBENCH_OBJS=()
if grep -q '#include\s*<polybench.h>\|#include\s*"polybench.h"' "$HARNESS_INPUT"; then
  # Find polybench.c on the same -I path the harness was given.
  POLYBENCH_C=""
  for arg in "${GCC_PASSTHROUGH[@]}"; do
    case "$arg" in
      -I*)
        dir=${arg#-I}
        if [ -f "$dir/polybench.c" ]; then POLYBENCH_C="$dir/polybench.c"; break; fi ;;
    esac
  done
  if [ -n "$POLYBENCH_C" ]; then
    echo "         + polybench utility from $POLYBENCH_C"
    $CC -O2 "${GCC_PASSTHROUGH[@]}" -c "$POLYBENCH_C" -o $WORK/polybench.o
    POLYBENCH_OBJS=("$WORK/polybench.o")
  fi
fi

CUSTOM_CUDA_OBJS=()
CUSTOM_CUDA_OBJ_LIST="${POLYGEIST_CUSTOM_CUDA_OBJS:-${POLYGEIST_CUSTOM_CUDA_OBJ:-}}"
if [ -n "$CUSTOM_CUDA_OBJ_LIST" ]; then
  read -r -a CUSTOM_CUDA_OBJS <<< "$CUSTOM_CUDA_OBJ_LIST"
  for obj in "${CUSTOM_CUDA_OBJS[@]}"; do
    [ -f "$obj" ] || {
      echo "ERROR: custom CUDA object $obj not found" >&2
      exit 1
    }
  done
  echo "         + custom CUDA object(s): ${CUSTOM_CUDA_OBJS[*]}"
fi

# ─── Step 9: link ───────────────────────────────────────────────────────
echo "  [9/9] link → $OUT"
$CC -O2 \
  $WORK/kernel.o $WORK/wrapper.o $WORK/harness.o $WORK/rt.o \
  $WORK/mlir_runner_utils.o \
  "${POLYBENCH_OBJS[@]}" \
  "${CUSTOM_CUDA_OBJS[@]}" \
  $RT_LIBS \
  -o "$OUT"

echo ""
echo "═══ build complete ═══"
file "$OUT" || true
