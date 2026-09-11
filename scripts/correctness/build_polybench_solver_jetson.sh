#!/bin/bash
# Cross-build source PolyBench/C Cholesky or Trisolv through the external
# cuSOLVER/cuBLAS routes, with a high-precision source-reference harness.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

KERNEL="${1:-}"
OUT="${2:-$PWD/polybench_${KERNEL}_jetson}"
DATASET="${POLYGEIST_SOLVER_DATASET:-MEDIUM}"
case "$DATASET" in
  MINI) N=40 ;;
  SMALL) N=120 ;;
  MEDIUM) N=400 ;;
  LARGE) N=2000 ;;
  EXTRALARGE) N=4000 ;;
  *) echo "ERROR: unsupported POLYGEIST_SOLVER_DATASET=$DATASET" >&2; exit 2 ;;
esac
PB="$REPO_ROOT/tools/cgeist/Test/polybench"
UTIL="$PB/utilities"

case "$KERNEL" in
  cholesky)
    SOURCE="$PB/linear-algebra/solvers/cholesky/cholesky.c"
    INCLUDE="$PB/linear-algebra/solvers/cholesky"
    HARNESS="$SCRIPT_DIR/polybench_cholesky_validation.c"
    FUNCTION=kernel_cholesky
    EXPECT=cusolverDnDpotrfLowerRowMajor
    ;;
  trisolv)
    SOURCE="$PB/linear-algebra/solvers/trisolv/trisolv.c"
    INCLUDE="$PB/linear-algebra/solvers/trisolv"
    HARNESS="$SCRIPT_DIR/polybench_trisolv_validation.c"
    FUNCTION=kernel_trisolv
    EXPECT=cublasDtrsvLowerRowMajor
    ;;
  *)
    echo "usage: $0 cholesky|trisolv [output]" >&2
    exit 2
    ;;
esac

WORK="$(mktemp -d)"
if [ "${POLYGEIST_KEEP_WORK:-0}" != "0" ]; then
  echo "[polybench-solver] keeping workdir: $WORK"
else
  trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK/export"

echo "[polybench-solver] $KERNEL n=$N through external library selection"
PYTHON="${PYTHON:-/usr/bin/python3}" \
POLYGEIST_WRAP_KERNEL_PIPELINE=0 \
POLYGEIST_EXPORT_OBJECT_DIR="$WORK/export" \
  "$SCRIPT_DIR/polygeist_build.sh" --target=jetson --no-debuf \
    --function="$FUNCTION" --harness="$HARNESS" -o "$OUT" "$SOURCE" \
    -D"${DATASET}_DATASET" -DDATA_TYPE_IS_DOUBLE \
    -DPOLYGEIST_SOLVER_N="$N" -Dstatic= -I"$UTIL" -I"$INCLUDE"

case "$KERNEL" in
  cholesky) ABI_CALL=polygeist_cusolver_dpotrf_lower_row_major ;;
  trisolv) ABI_CALL=polygeist_cublas_dtrsv_lower_row_major ;;
esac
CALLS="$(grep -c "call @${ABI_CALL}" "$WORK/export/abi.mlir" || true)"
if [ "$CALLS" -ne 1 ]; then
  echo "ERROR: expected one $EXPECT call; got $CALLS" >&2
  exit 1
fi
echo "[polybench-solver] complete: $OUT ($EXPECT)"
file "$OUT"
