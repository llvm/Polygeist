#!/bin/bash
# Sweep the PolyBench-style extracted NPB kernels through the raise pipeline.
# Each kernel is a single .c file in third_party/NPB-polybenchified/ that
# takes its arrays as parameters (no module-level static globals).
set +e
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"
DIR=$REPO_ROOT/third_party/NPB-polybenchified
OUT=/tmp/npb_extracted_sweep
mkdir -p $OUT

# Format: <tag>  <fn>
KERNELS=(
  "bt-add       bt_add"
  "ft-evolve    ft_evolve"
  "lu-l2norm    lu_l2norm"
  "mg-psinv     mg_psinv"
  "mg-resid     mg_resid"
  "mg-norm2u3   mg_norm2u3"
  "mg-rprj3     mg_rprj3"
)

printf '%-12s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s   %s\n' \
  kernel CG_LG CG_AF CG_SF RS_LG RS_AF RS_SF DB_LG DB_AF DB_SF status
echo "----------------------------------------------------------------------------------"

for entry in "${KERNELS[@]}"; do
  read tag fn <<<"$entry"
  src="$DIR/${tag//-/_}.c"
  [ ! -f "$src" ] && { printf '%-12s missing %s\n' "$tag" "$src"; continue; }

  timeout 60 cgeist "$src" --function=$fn --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine -fPIC -S -o $OUT/${tag}.mlir 2>$OUT/${tag}.cgeist.err
  if [ ! -s $OUT/${tag}.mlir ]; then
    printf '%-12s   --   --   --      --   --   --      --   --   --   CGEIST_FAIL\n' "$tag"; continue
  fi
  CG_LG=$(grep -c "linalg.generic" $OUT/${tag}.mlir 2>/dev/null); CG_LG=${CG_LG:-0}
  CG_AF=$(grep -c "affine.for" $OUT/${tag}.mlir 2>/dev/null); CG_AF=${CG_AF:-0}
  CG_SF=$(grep -cE "scf\.(for|while)" $OUT/${tag}.mlir 2>/dev/null); CG_SF=${CG_SF:-0}

  timeout 60 polygeist-opt --select-func=func-name=$fn \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${tag}.mlir -o $OUT/${tag}.raised.mlir 2>$OUT/${tag}.raise.err
  if [ ! -s $OUT/${tag}.raised.mlir ]; then
    printf '%-12s %5s %5s %5s     --   --   --      --   --   --   RAISE_FAIL\n' \
      "$tag" "$CG_LG" "$CG_AF" "$CG_SF"; continue
  fi
  RS_LG=$(grep -c "linalg.generic" $OUT/${tag}.raised.mlir 2>/dev/null); RS_LG=${RS_LG:-0}
  RS_AF=$(grep -c "affine.for" $OUT/${tag}.raised.mlir 2>/dev/null); RS_AF=${RS_AF:-0}
  RS_SF=$(grep -cE "scf\.(for|while)" $OUT/${tag}.raised.mlir 2>/dev/null); RS_SF=${RS_SF:-0}

  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
    $OUT/${tag}.raised.mlir -o $OUT/${tag}.debuf.mlir 2>$OUT/${tag}.debuf.err
  if [ ! -s $OUT/${tag}.debuf.mlir ]; then
    printf '%-12s %5s %5s %5s   %5s %5s %5s      --   --   --   DEBUF_FAIL\n' \
      "$tag" "$CG_LG" "$CG_AF" "$CG_SF" "$RS_LG" "$RS_AF" "$RS_SF"; continue
  fi
  DB_LG=$(grep -c "linalg.generic" $OUT/${tag}.debuf.mlir 2>/dev/null); DB_LG=${DB_LG:-0}
  DB_AF=$(grep -c "affine.for" $OUT/${tag}.debuf.mlir 2>/dev/null); DB_AF=${DB_AF:-0}
  DB_SF=$(grep -cE "scf\.(for|while)" $OUT/${tag}.debuf.mlir 2>/dev/null); DB_SF=${DB_SF:-0}

  if [ "$DB_LG" -gt 0 ] && [ "$DB_AF" -eq 0 ] && [ "$DB_SF" -eq 0 ]; then
    status=FULL_LIFT
  elif [ "$DB_LG" -gt 0 ]; then
    status=PARTIAL_LIFT
  else
    status=NO_LIFT
  fi
  printf '%-12s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s   %s\n' \
    "$tag" "$CG_LG" "$CG_AF" "$CG_SF" "$RS_LG" "$RS_AF" "$RS_SF" \
    "$DB_LG" "$DB_AF" "$DB_SF" "$status"
done
