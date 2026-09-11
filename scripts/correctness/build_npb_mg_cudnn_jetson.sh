#!/bin/bash
# Cross-build the original NPB3.3-SER-C MG Class S application with resid,
# psinv, and rprj3 raised separately and composed through external cuDNN.
# Nothing is compiled on the Jetson.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

OUT="${1:-$PWD/npb_mg_cudnn_jetson}"
NPB_ROOT="$REPO_ROOT/third_party/ginsbach-snu-npb/NPB3.3-SER-C"
MG_SOURCE="$NPB_ROOT/MG/mg.c"
COMMON="$NPB_ROOT/common"
PARAMS="$REPO_ROOT/issues/ginsbach_asplos18/npb_mg_class_s"
CUDA_CROSS="${CUDA_CROSS:-/usr/local/cuda-12.6/targets/sbsa-linux}"
CUDNN_CROSS_LIB="${CUDNN_CROSS_LIB:-/usr/lib/aarch64-linux-gnu}"
CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"

for required in "$MG_SOURCE" "$PARAMS/npbparams.h"; do
  [ -f "$required" ] || { echo "ERROR: missing $required" >&2; exit 1; }
done

WORK="$(mktemp -d)"
if [ "${POLYGEIST_KEEP_WORK:-0}" != "0" ]; then
  echo "[npb-mg] keeping workdir: $WORK"
else
  trap 'rm -rf "$WORK"' EXIT
fi

for source in print_results randdp c_timers wtime; do
  "$CC" -O2 -ffunction-sections -fdata-sections -I"$COMMON" \
    -c "$COMMON/$source.c" -o "$WORK/$source.o"
done

for function in resid psinv rprj3; do
  object_dir="$WORK/$function"
  echo "[npb-mg] raising $function"
  POLYGEIST_MINIMAL_CUDNN_RUNTIME=1 \
  POLYGEIST_EXPORT_OBJECT_DIR="$object_dir" \
  POLYGEIST_SKIP_LINK=1 \
    "$SCRIPT_DIR/polygeist_build.sh" --target=jetson \
      --function="$function" -o "$WORK/unused-$function" "$MG_SOURCE" \
      -Dstatic= -I"$PARAMS" -I"$NPB_ROOT/MG" -I"$COMMON"
  launches="$(grep -c 'kernel.launch @cudnnStencil3DSymmetric_f64_memref' \
    "$object_dir/matched.mlir" || true)"
  [ "$launches" -eq 1 ] || {
    echo "ERROR: $function produced $launches MG cuDNN launches, expected 1" >&2
    exit 1
  }
done

# The resid object supplies the original application harness. Its selected
# symbol is already weak; weaken the other two C definitions so all three
# strong lifted functions replace them in the final binary.
cp "$WORK/resid/harness.o" "$WORK/mg_harness.o"
aarch64-linux-gnu-objcopy --weaken-symbol=psinv --weaken-symbol=rprj3 \
  "$WORK/mg_harness.o"

"$CC" -O2 -Wl,--gc-sections -o "$OUT" \
  "$WORK/resid/kernel.o" "$WORK/psinv/kernel.o" "$WORK/rprj3/kernel.o" \
  "$WORK/mg_harness.o" "$WORK/resid/rt.o" \
  "$WORK/resid/mlir_runner_utils.o" \
  "$WORK/print_results.o" "$WORK/randdp.o" \
  "$WORK/c_timers.o" "$WORK/wtime.o" \
  -L"$CUDA_CROSS/lib" -L"$CUDA_CROSS/lib/stubs" -L"$CUDNN_CROSS_LIB" \
  -lcudnn -lcublasLt -lcublas -lcudart -lm -lpthread -ldl \
  -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/home/nvidia/polygeist_cuda_libs

echo "[npb-mg] complete: resid + psinv + rprj3 -> external cuDNN"
file "$OUT"
