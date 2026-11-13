#!/bin/bash
set +e
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt

OUT_DIR="/tmp/lowering_test"
mkdir -p "$OUT_DIR"

LOWERING_PIPE="--expand-strided-metadata \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --convert-math-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts"

# Reuse the kernel list from /tmp/run_polybench.sh
KERNELS=(
  "correlation" "covariance" "durbin" "cholesky" "gramschmidt"
  "lu" "ludcmp" "trisolv" "gemm" "syr2k" "syrk" "gesummv" "symm"
  "trmm" "gemver" "bicg" "doitgen" "atax" "mvt" "2mm" "3mm"
  "heat-3d" "jacobi-2d" "jacobi-1d" "adi" "fdtd-2d" "seidel-2d"
  "floyd-warshall" "deriche" "nussinov"
)

pass=0
fail_lower=0
fail_llvm=0

for k in "${KERNELS[@]}"; do
  src="/tmp/polybench_new/${k}_linalg.mlir"
  if [ ! -f "$src" ]; then echo "$k: NO_INPUT"; continue; fi
  
  step1="$OUT_DIR/${k}_step1.mlir"
  step2="$OUT_DIR/${k}_step2.mlir"
  log="$OUT_DIR/${k}.log"
  
  # Step 1: lower polygeist.submap to standard MLIR
  polygeist-opt --lower-polygeist-submap "$src" -o "$step1" 2> "$log"
  if [ ! -s "$step1" ]; then echo "$k: LOWER_SUBMAP_FAIL"; fail_lower=$((fail_lower+1)); continue; fi
  
  # Check no polygeist ops remain (be precise; "polygeist.target-cpu" in attrs is OK)
  remain=$(grep -cE "polygeist\.(submap|submapInverse|trivialuse|alternatives|barrier|kernelinfo|cache|noop|gpu|getfunc|stream)" "$step1" 2>/dev/null || echo 0)
  if [ "$remain" -gt 0 ]; then
    echo "$k: PARTIAL_LOWER (${remain} polygeist ops remain)"
    fail_lower=$((fail_lower+1))
    continue
  fi
  
  # Step 2: standard MLIR lowering to LLVM dialect
  $MLIR_OPT $LOWERING_PIPE "$step1" -o "$step2" 2>> "$log"
  if [ ! -s "$step2" ]; then echo "$k: LLVM_LOWER_FAIL"; fail_llvm=$((fail_llvm+1)); continue; fi
  
  echo "$k: OK"
  pass=$((pass+1))
done

echo "---"
echo "Summary: $pass passed, $fail_lower submap-lower failed, $fail_llvm llvm-lower failed"
