#!/bin/bash
# gemm_cublas_jetson.sh — produce a Jetson-ready aarch64 binary of gemm
# routed through our matcher + cuBLAS-ABI lowering.
#
# Mirrors the structure of gemm_cublas_e2e.sh, but:
#   * Stops before the local execute/diff (no x86 run; the binary is for ARM).
#   * Cross-compiles polybench's gemm.c + polybench.c here with the right
#     POLYBENCH defines, then hands them as pre-built .o files to
#     build_jetson.sh.
#   * Wraps kernel_gemm with the timing wrapper at gemm_jetson_wrapper.c so
#     each call prints "POLYGEIST_TIMING: kernel_gemm ... <ms> ms" to stderr
#     when run on the Jetson.
#
# Usage:
#   ./gemm_cublas_jetson.sh [DATASET]
# DATASET defaults to MINI; pass STANDARD or LARGE for bigger problems.
#
# Output: /tmp/gemm_cublas_jetson_build/gemm_jetson  (aarch64 ELF, ~20 KB)
# Then scp to Jetson and run.

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

DATASET=${1:-MINI}
case "$DATASET" in
  MINI|SMALL|STANDARD|LARGE|EXTRALARGE) ;;
  *) echo "ERROR: DATASET must be one of MINI|SMALL|STANDARD|LARGE|EXTRALARGE" >&2; exit 1 ;;
esac

OUT=/tmp/gemm_cublas_jetson_build
mkdir -p $OUT

POLYBENCH_DIR=/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench
UTIL=$POLYBENCH_DIR/utilities
GEMM_DIR=$POLYBENCH_DIR/linear-algebra/blas/gemm
SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness
RT=/home/arjaiswal/Polygeist/runtime

# Harness CFLAGS for cross-compiling polybench's gemm.c + polybench.c.
HARNESS_CFLAGS=(-O3 -I"$UTIL" -I"$GEMM_DIR"
    -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS
    -D${DATASET}_DATASET
    -Dstatic= -DPOLYBENCH_USE_C99_PROTO)

# ─── Step 1: produce the ABI-lowered MLIR (reuse gemm_cublas_e2e.sh artifacts) ─
ABI_MLIR=/tmp/gemm_cublas_test/gemm_abi.mlir
if [ ! -s "$ABI_MLIR" ]; then
  echo "[gemm-jetson] producing ABI-lowered MLIR via gemm_cublas_e2e.sh..."
  bash $SCRIPTS/gemm_cublas_e2e.sh >/tmp/gemm_cublas_test/local_e2e.log 2>&1
fi
if [ ! -s "$ABI_MLIR" ]; then
  echo "ERROR: $ABI_MLIR missing after gemm_cublas_e2e.sh" >&2
  exit 1
fi

# ─── Step 2: cross-compile polybench harness pieces for aarch64 ────────────
echo "[gemm-jetson] cross-compiling polybench gemm.c + polybench.c (dataset=$DATASET)"
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$GEMM_DIR/gemm.c" -o $OUT/gemm_full.o
aarch64-linux-gnu-objcopy --weaken-symbol=kernel_gemm $OUT/gemm_full.o $OUT/gemm_nokernel.o
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$UTIL/polybench.c" -o $OUT/polybench.o

# ─── Step 3: invoke build_jetson.sh with all the harness pieces ────────────
# Pass:
#   * gemm_jetson_wrapper.c  — adds timing around the lowered kernel
#   * gemm_nokernel.o         — polybench gemm.c with kernel_gemm weakened
#   * polybench.o             — polybench timing / IO helpers
echo "[gemm-jetson] invoking build_jetson.sh"
bash $SCRIPTS/build_jetson.sh \
    "$ABI_MLIR" \
    "$OUT/gemm_jetson" \
    "$SCRIPTS/gemm_jetson_wrapper.c" \
    "$OUT/gemm_nokernel.o" \
    "$OUT/polybench.o"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Binary ready: $OUT/gemm_jetson"
echo "Dataset:      ${DATASET}_DATASET (problem size baked into polybench.o)"
echo ""
echo "Ship + run (once SSH is sorted):"
echo "  scp $OUT/gemm_jetson <user>@<jetson>:/tmp/"
echo "  ssh <user>@<jetson> 'chmod +x /tmp/gemm_jetson && /tmp/gemm_jetson 2>&1'"
echo ""
echo "Look for 'POLYGEIST_TIMING:' lines on stderr for per-call ms."
echo "═══════════════════════════════════════════════════════════════════════"
