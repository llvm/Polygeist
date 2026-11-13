#!/bin/bash
# Every harnessed MFEM kernel on the CPU: build each normalised kernel through
# the LoopLibrarian pipeline with network composition (build_whole_operator.sh,
# in parallel), then validate every output element and time it against plain
# C (run_kernel_cpu.sh, sequentially so timings do not interfere).
#
# Usage: [COMPOSE_OPT=<polygeist-opt>] [JOBS=6] [REPS=15] [ONLY="mass_2d ..."]
#        run_all_cpu.sh [BUILD_ROOT] [OUT_DIR]
set -uo pipefail
REPO=/home/arjaiswal/Polygeist
ROOT=${1:-/tmp/mfem_cpu/all}
OUT=${2:-$REPO/issues/mfem_cpu_experiment/all_kernels}
JOBS=${JOBS:-6}
HERE=$REPO/scripts/correctness/mfem_cpu
N=$REPO/issues/mfem_c_kernels/normalized
export COMPOSE_OPT=${COMPOSE_OPT:-$REPO/build-v2/bin/polygeist-opt}
mkdir -p "$ROOT" "$OUT"

# name  harness-define  function  normalised-source
KERNELS=$(cat <<'EOF'
mass_2d BENCH_MASS_2D mfem_pa_mass_apply_2d_stage_sliced mass_stage_sliced.c
mass_3d BENCH_MASS_3D mfem_pa_mass_apply_3d_stage_sliced mass_stage_sliced.c
diffusion_2d BENCH_DIFFUSION_2D mfem_pa_diffusion_apply_2d_stage_sliced diffusion_stage_sliced.c
diffusion_3d BENCH_DIFFUSION_3D mfem_pa_diffusion_apply_3d_stage_sliced diffusion_stage_sliced.c
convection_2d BENCH_CONVECTION_2D mfem_pa_convection_apply_2d_stage_sliced convection_stage_sliced.c
convection_3d BENCH_CONVECTION_3D mfem_pa_convection_apply_3d_stage_sliced convection_stage_sliced.c
curlcurl_2d BENCH_CURLCURL_2D mfem_pa_curlcurl_apply_2d_stage_sliced de_rham2_stage_sliced.c
curlcurl_3d BENCH_CURLCURL_3D mfem_pa_curlcurl_apply_3d_stage_sliced curlcurl3_stage_sliced.c
divdiv_2d BENCH_DIVDIV_2D mfem_pa_divdiv_apply_2d_stage_sliced de_rham2_stage_sliced.c
divdiv_3d BENCH_DIVDIV_3D mfem_pa_divdiv_apply_3d_stage_sliced divdiv3_stage_sliced.c
interp_value_2d BENCH_INTERP_VALUE_2D mfem_interp_value_2d_scratch_sliced value_scratch_sliced.c
interp_value_3d BENCH_INTERP_VALUE_3D mfem_interp_value_3d_scratch_sliced value_scratch_sliced.c
integrate_value_2d BENCH_INTEGRATE_VALUE_2D mfem_integrate_value_2d_scratch_sliced value_scratch_sliced.c
integrate_value_3d BENCH_INTEGRATE_VALUE_3D mfem_integrate_value_3d_scratch_sliced value_scratch_sliced.c
interp_grad_2d BENCH_INTERP_GRAD_2D mfem_interp_grad_2d_stage_sliced gradient_stage_sliced.c
interp_grad_3d BENCH_INTERP_GRAD_3D mfem_interp_grad_3d_stage_sliced gradient_stage_sliced.c
integrate_grad_2d BENCH_INTEGRATE_GRAD_2D mfem_integrate_grad_2d_stage_sliced gradient_stage_sliced.c
integrate_grad_3d BENCH_INTEGRATE_GRAD_3D mfem_integrate_grad_3d_stage_sliced gradient_stage_sliced.c
EOF
)
if [ -n "${ONLY:-}" ]; then
  KERNELS=$(echo "$KERNELS" | awk -v only=" $ONLY " 'index(only, " " $1 " ")')
fi

echo "== build ($JOBS in parallel; logs in $ROOT/<name>.log)"
echo "$KERNELS" | xargs -P "$JOBS" -L 1 bash -c '
  name=$0; def=$1; fn=$2; src=$3
  if bash "'"$HERE"'/build_whole_operator.sh" "'"$N"'/$src" "$fn" "$def" "'"$ROOT"'/$name" \
       > "'"$ROOT"'/$name.log" 2>&1; then echo "built $name"; else echo "BUILD FAILED $name"; fi'

echo "== validate and time"
echo "$KERNELS" | while read -r name def fn src; do
  [ -f "$ROOT/$name/obj/rt.o" ] || { echo "skip $name (no objects)"; continue; }
  mkdir -p "$OUT/$name"
  if OUT="$OUT/$name" BENCH_DEF=$def KERNEL_SRC="$N/$src" REF_INTERP=0 \
       bash "$HERE/run_kernel_cpu.sh" "$ROOT/$name/obj" "$ROOT/$name/work" \
       > "$OUT/$name/run.log" 2>&1 < /dev/null; then echo "ran $name"; else echo "RUN FAILED $name"; fi
done

/usr/bin/python3 "$HERE/summarize_all.py" "$ROOT" "$OUT" | tee "$OUT/SUMMARY.txt"
