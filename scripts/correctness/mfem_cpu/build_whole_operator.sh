#!/bin/bash
# Build an MFEM kernel so that an additive final contraction (e.g. mass-2D's
# `Y += Bt * tx`) is absorbed into the cuTensorNet network.  The composition
# pass normalises zero-stride reduction outputs with the existing
# NormalizeTensorReductionOutputSubmap pattern before tracing sinks.
# polygeist_build.sh is not modified: we inject the canonical defns into its
# matcher output and compose with COMPOSE_OPT (a polygeist-opt that has the
# change), then resume the script from the composed artifact with
# --semantic-mlir.
#
# Usage: build_whole_operator.sh <normalized.c> <function> <bench-define> [OUTDIR]
#   e.g. build_whole_operator.sh issues/mfem_c_kernels/normalized/mass_stage_sliced.c \
#          mfem_pa_mass_apply_2d_stage_sliced BENCH_MASS_2D /tmp/mfem_cpu/acc
set -euo pipefail
REPO=/home/arjaiswal/Polygeist
SRC=$1; FN=$2; BENCH_DEF=$3; W=${4:-/tmp/mfem_cpu/acc}
NE=${NE:-1024}
source "$REPO/envsetup.sh" >/dev/null 2>&1
mkdir -p "$W"
HARNESS=$REPO/issues/mfem_c_kernels/benchmarks/mfem_raised_pa_bench.c

# 1. normal build to obtain the matcher output (matched.mlir)
POLYGEIST_CGEIST_FLAGS="-DMFEM_BENCH_NE=$NE" POLYGEIST_BUILD_WORK_DIR="$W/base_work" \
  bash "$REPO/scripts/correctness/polygeist_build.sh" --target=host --function="$FN" \
  --harness="$HARNESS" -o "$W/base_bin" "$SRC" -DMFEM_BENCH_NE=$NE -D"$BENCH_DEF" > "$W/base_build.log" 2>&1

# 2. inject canonical defns and compose networks, as polygeist_build.sh step 4
KLIB=$REPO/generic_solver/kernel_library_phase2.mlir
DEFNS=$(sed -n '/^module {$/,/^}$/p' "$KLIB" | sed '1d; $d')
awk -v defns="$DEFNS" '/^module attributes/ && !done { print; print defns; done=1; next } { print }' \
  "$W/base_work/matched.mlir" > "$W/with_defns.mlir"
"${COMPOSE_OPT:-polygeist-opt}" --compose-cutensornet-networks "$W/with_defns.mlir" -o "$W/with_defns_composed.mlir"
grep -o "kernel.launch @[A-Za-z0-9_]*" "$W/with_defns_composed.mlir" | sort | uniq -c

# 3. resume the normal build from the composed artifact (ABI lowering, link)
rm -rf "$W/obj" "$W/work"
POLYGEIST_CGEIST_FLAGS="-DMFEM_BENCH_NE=$NE" POLYGEIST_EXPORT_OBJECT_DIR="$W/obj" \
  POLYGEIST_BUILD_WORK_DIR="$W/work" \
  bash "$REPO/scripts/correctness/polygeist_build.sh" --target=host --function="$FN" \
  --harness="$HARNESS" --semantic-mlir="$W/with_defns_composed.mlir" -o "$W/bin" "$SRC" \
  -DMFEM_BENCH_NE=$NE -D"$BENCH_DEF" > "$W/build.log" 2>&1
echo "objects in $W/obj, work dir $W/work"
