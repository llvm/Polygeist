#!/bin/bash
# One MFEM kernel on the CPU: plain C vs LoopLibrarian-matched program whose
# tensor-network/contraction calls run through runtime/polygeist_tensornet_cpu.c
# (backends: loops | openblas | xsmm).  Both sides use the repo's clang with the
# same flags, the same benchmark harness source and the same machine.
#
# Inputs: objects exported by polygeist_build.sh --target=host
#   (POLYGEIST_EXPORT_OBJECT_DIR=$OBJ, POLYGEIST_BUILD_WORK_DIR=$WORK).
# Usage: [BENCH_DEF=BENCH_MASS_2D] [KERNEL_SRC=<normalized.c>] [REF_INTERP=1]
#        run_kernel_cpu.sh [OBJ_DIR] [WORK_DIR]
set -euo pipefail
REPO=/home/arjaiswal/Polygeist
OBJ=${1:-/tmp/mfem_cpu/obj}
WORK=${2:-/tmp/mfem_cpu/work}
CLANG=$REPO/llvm-project/build/bin/clang
XSMM=${XSMM:-/home/arjaiswal/third_party_ext/libxsmm}
OUT=${OUT:-$REPO/issues/mfem_cpu_experiment}
BIN=$OUT/bin
NE=1024
ITERS=${ITERS:-200}
REPS=${REPS:-5}
FLAGS=(-O3 -march=native)
BENCH=$REPO/issues/mfem_c_kernels/benchmarks
BENCH_DEF=${BENCH_DEF:-BENCH_MASS_2D}
KERNEL_SRC=${KERNEL_SRC:-$REPO/issues/mfem_c_kernels/normalized/mass_stage_sliced.c}
REF_INTERP=${REF_INTERP:-1}
mkdir -p "$BIN"
# The normalised 3-D sources keep per-element scratch arrays on the stack
# (6-10 MB at NE=1024), above the usual 8 MB default.  Same limit for every
# variant.
ulimit -s 1048576

echo "== build"
$CLANG "${FLAGS[@]}" -DPOLYGEIST_TN_HAVE_OPENBLAS -DPOLYGEIST_TN_HAVE_XSMM -I"$XSMM/include" \
  -c "$REPO/runtime/polygeist_tensornet_cpu.c" -o "$BIN/tensornet_cpu.o"
objcopy --weaken-symbol=polygeist_cutensornet_network_f64 \
        --weaken-symbol=polygeist_cutensornet_network_f64_device \
        --weaken-symbol=polygeist_cutensornet_contraction2_f64 \
        --weaken-symbol=polygeist_cutensornet_contraction2_f64_device \
        "$OBJ/rt.o" "$BIN/rt_weak.o"
$CLANG "${FLAGS[@]}" -c "$WORK/kernel.ll" -o "$BIN/kernel.o"
for it in "$ITERS" 2; do
  $CLANG "${FLAGS[@]}" -D"$BENCH_DEF" -DMFEM_BENCH_NE=$NE -DBENCH_ITERS="$it" \
    -c "$BENCH/mfem_raised_pa_bench.c" -o "$BIN/harness_$it.o"
done
LIBS=("$XSMM/lib/libxsmm.a" "$XSMM/lib/libxsmmgen.a" -lopenblas -lpthread -ldl -lm)
RAISED_OBJS=("$BIN/kernel.o" "$OBJ/wrapper.o" "$BIN/rt_weak.o" "$OBJ/mlir_runner_utils.o" "$BIN/tensornet_cpu.o")
$CLANG "${FLAGS[@]}" "${RAISED_OBJS[@]}" "$BIN/harness_$ITERS.o" "${LIBS[@]}" -o "$BIN/raised"
# Reference interpreter (the runtime the paper's CPU numbers used), few iterations.
if [ "$REF_INTERP" = 1 ]; then
  $CLANG "${FLAGS[@]}" "$BIN/kernel.o" "$OBJ/wrapper.o" "$OBJ/rt.o" "$OBJ/mlir_runner_utils.o" \
    "$BIN/harness_2.o" -lm -lpthread -o "$BIN/raised_reference_interp"
fi
# Plain C: same harness source, normalised MFEM source, same compiler and flags.
$CLANG "${FLAGS[@]}" -D"$BENCH_DEF" -DMFEM_BENCH_NE=$NE -DBENCH_ITERS="$ITERS" \
  "$BENCH/mfem_raised_pa_bench.c" "$KERNEL_SRC" \
  -lm -o "$BIN/vanilla"
# Full-output validation: every output element vs the normalised C, same inputs.
$CLANG "${FLAGS[@]}" -D"$BENCH_DEF" -DMFEM_BENCH_NE=$NE -c "$BENCH/mfem_full_output_validation.c" \
  -o "$BIN/validation.o"
$CLANG "${FLAGS[@]}" "${RAISED_OBJS[@]}" "$BIN/validation.o" "${LIBS[@]}" -o "$BIN/validate"

echo "== validate"
for be in loops openblas xsmm; do
  echo "-- backend=$be"
  POLYGEIST_TENSORNET_BACKEND=$be "$BIN/validate" | tail -3
done
echo "-- contraction order chosen (first call)"
POLYGEIST_TENSORNET_VERBOSE=1 POLYGEIST_TENSORNET_BACKEND=loops "$BIN/validate" 2>&1 >/dev/null | head -8

echo "== time (runtime_us per operator application; $REPS processes x $ITERS iterations)"
run() { # label cmd...
  local label=$1; shift
  local vals=()
  for r in $(seq "$REPS"); do
    vals+=("$("$@" | sed -n 's/.*runtime_us=\([0-9.]*\).*/\1/p')")
  done
  printf '%s %s\n' "$label" "${vals[*]}" | tee -a "$OUT/timings.txt"
}
: > "$OUT/timings.txt"
run vanilla_c "$BIN/vanilla"
for be in loops openblas xsmm; do
  run "raised_$be" env POLYGEIST_TENSORNET_BACKEND=$be "$BIN/raised"
done
run raised_xsmm_noslice env POLYGEIST_TENSORNET_SLICE=0 POLYGEIST_TENSORNET_BACKEND=xsmm "$BIN/raised"
if [ "$REF_INTERP" = 1 ]; then
  REPS=1 run raised_reference_interpreter "$BIN/raised_reference_interp"
fi
