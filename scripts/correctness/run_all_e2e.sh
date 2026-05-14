#!/bin/bash
# Run e2e for every PolyBench kernel that lowers clean through our pass.
# Reports PASS / FAIL_<stage> for each.
set +e
SCRIPT=/home/arjaiswal/Polygeist/scripts/correctness/run_kernel_e2e.sh
PB=/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench
MODE="${1:-}"   # "" or "--debuf"

# (relative_dir, kernel_short_name) for the 17 lowering-clean kernels.
declare -a KERNELS=(
  "linear-algebra/blas/gemm gemm"
  "linear-algebra/blas/syr2k syr2k"
  "linear-algebra/blas/syrk syrk"
  "linear-algebra/blas/gesummv gesummv"
  "linear-algebra/blas/gemver gemver"
  "linear-algebra/blas/symm symm"
  "linear-algebra/blas/trmm trmm"
  "linear-algebra/kernels/bicg bicg"
  "linear-algebra/kernels/atax atax"
  "linear-algebra/kernels/mvt mvt"
  "linear-algebra/kernels/2mm 2mm"
  "linear-algebra/kernels/3mm 3mm"
  "linear-algebra/kernels/doitgen doitgen"
  "linear-algebra/solvers/cholesky cholesky"
  "linear-algebra/solvers/gramschmidt gramschmidt"
  "linear-algebra/solvers/lu lu"
  "linear-algebra/solvers/trisolv trisolv"
  "stencils/heat-3d heat-3d"
  "stencils/jacobi-2d jacobi-2d"
  "stencils/jacobi-1d jacobi-1d"
  "stencils/fdtd-2d fdtd-2d"
  "medley/floyd-warshall floyd-warshall"
  "medley/deriche deriche"
  "medley/nussinov nussinov"
  "datamining/correlation correlation"
  "datamining/covariance covariance"
)

pass=0
fail=0
for entry in "${KERNELS[@]}"; do
  read -r reldir short <<< "$entry"
  # Grab the first PASS/FAIL/PARTIAL marker emitted by the per-kernel
  # script (those are followed by diff context that 'tail -1' would catch).
  out=$($SCRIPT "$PB/$reldir" "$short" $MODE 2>&1 | grep -E "PASS|FAIL|PARTIAL|MISSING" | head -1)
  [ -z "$out" ] && out="$short: NO_RESULT"
  echo "$out"
  if [[ "$out" == *PASS* ]]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done
echo "---"
echo "Total: $pass pass, $fail fail"
