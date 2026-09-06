#!/bin/bash
# Cross-build the source-faithful NPB3.3-SER-C BT Class S application.
# Compilation is local; the resulting AArch64 executable is ready for Orin.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

OUT="${1:-$PWD/npb_bt_class_s_jetson}"
NPB_ROOT="$REPO_ROOT/third_party/ginsbach-snu-npb/NPB3.3-SER-C"
BT="$NPB_ROOT/BT"
COMMON="$NPB_ROOT/common"
PARAMS="$REPO_ROOT/issues/ginsbach_asplos18/npb_bt_class_s"
CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for source in bt initialize exact_solution exact_rhs set_constants adi rhs \
              x_solve y_solve solve_subs z_solve add error verify; do
  "$CC" -O2 -ffunction-sections -fdata-sections \
    -I"$PARAMS" -I"$BT" -I"$COMMON" -c "$BT/$source.c" \
    -o "$WORK/$source.o"
done
for source in print_results c_timers wtime; do
  "$CC" -O2 -ffunction-sections -fdata-sections -I"$COMMON" \
    -c "$COMMON/$source.c" -o "$WORK/$source.o"
done

"$CC" -O2 -Wl,--gc-sections -o "$OUT" "$WORK"/*.o -lm
file "$OUT"
