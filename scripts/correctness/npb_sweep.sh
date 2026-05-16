#!/bin/bash
# Sweep NPB-C benchmarks through the Polygeist raise pipeline.
#
# NPB-C is one big .c per benchmark (BT, LU, SP, MG, FT, CG, IS, EP),
# each containing many static kernel-shaped functions. Unlike PolyBench
# / MachSuite where each file has exactly one kernel, NPB references
# many module-level statics from each function — so `--select-func`
# (which strips global defs) yields invalid modules. We raise the
# whole .c file and report per-benchmark totals: # linalg.generic vs
# # residual affine.for / scf.for / scf.while.
set +e
source /home/arjaiswal/Polygeist/envsetup.sh
ROOT=/home/arjaiswal/Polygeist/third_party/NPB3.0-omp-C
COMMON=$ROOT/common
OUT=/tmp/npb_sweep
mkdir -p $OUT

BENCHES=(BT LU SP MG FT CG IS EP)

printf '%-6s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s   %s\n' \
  bench CG_LG CG_AF CG_SF RS_LG RS_AF RS_SF DB_LG DB_AF DB_SF status
echo "------------------------------------------------------------------------------"

for b in "${BENCHES[@]}"; do
  D=$ROOT/$b
  src=$D/$(echo $b | tr 'A-Z' 'a-z').c
  if [ ! -f "$src" ]; then
    printf '%-6s missing %s\n' "$b" "$src"; continue
  fi

  # Step 1: cgeist (whole module, all functions). NPB benchmarks are large
  # (BT/LU/SP each over 3000 LoC); give cgeist a generous budget.
  timeout 300 cgeist "$src" --function='*' --resource-dir=/usr/lib/clang/14 \
    -I$COMMON -I$D -Dstatic= \
    -DNPBVERSION='"3.0"' -DCOMPILETIME='"now"' \
    -DCS1='"cc"' -DCS2='"cc"' -DCS3='"-O3"' -DCS4='""' \
    -DCS5='""' -DCS6='""' -DCS7='""' \
    --raise-scf-to-affine -fPIC -S \
    -o $OUT/${b}.mlir 2>$OUT/${b}.cgeist.err
  if [ ! -s $OUT/${b}.mlir ]; then
    printf '%-6s   --   --   --      --   --   --      --   --   --   CGEIST_FAIL\n' "$b"
    continue
  fi
  CG_LG=$(grep -c "linalg.generic" $OUT/${b}.mlir 2>/dev/null); CG_LG=${CG_LG:-0}
  CG_AF=$(grep -c "affine.for" $OUT/${b}.mlir 2>/dev/null); CG_AF=${CG_AF:-0}
  CG_SF=$(grep -cE "scf\.(for|while)" $OUT/${b}.mlir 2>/dev/null); CG_SF=${CG_SF:-0}

  # Step 2: raise + lower-submap on the whole module.
  timeout 600 polygeist-opt \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${b}.mlir -o $OUT/${b}.raised.mlir 2>$OUT/${b}.raise.err
  if [ ! -s $OUT/${b}.raised.mlir ]; then
    printf '%-6s %5s %5s %5s     --   --   --      --   --   --   RAISE_FAIL\n' \
      "$b" "$CG_LG" "$CG_AF" "$CG_SF"
    continue
  fi
  RS_LG=$(grep -c "linalg.generic" $OUT/${b}.raised.mlir 2>/dev/null); RS_LG=${RS_LG:-0}
  RS_AF=$(grep -c "affine.for" $OUT/${b}.raised.mlir 2>/dev/null); RS_AF=${RS_AF:-0}
  RS_SF=$(grep -cE "scf\.(for|while)" $OUT/${b}.raised.mlir 2>/dev/null); RS_SF=${RS_SF:-0}

  # Step 3: debufferize (multi-root).
  timeout 180 polygeist-opt --linalg-debufferize=use-multi-root=true \
    $OUT/${b}.raised.mlir -o $OUT/${b}.debuf.mlir 2>$OUT/${b}.debuf.err
  if [ ! -s $OUT/${b}.debuf.mlir ]; then
    printf '%-6s %5s %5s %5s   %5s %5s %5s      --   --   --   DEBUF_FAIL\n' \
      "$b" "$CG_LG" "$CG_AF" "$CG_SF" "$RS_LG" "$RS_AF" "$RS_SF"
    continue
  fi
  DB_LG=$(grep -c "linalg.generic" $OUT/${b}.debuf.mlir 2>/dev/null); DB_LG=${DB_LG:-0}
  DB_AF=$(grep -c "affine.for" $OUT/${b}.debuf.mlir 2>/dev/null); DB_AF=${DB_AF:-0}
  DB_SF=$(grep -cE "scf\.(for|while)" $OUT/${b}.debuf.mlir 2>/dev/null); DB_SF=${DB_SF:-0}

  if [ "$DB_LG" -gt 0 ] && [ "$DB_AF" -eq 0 ] && [ "$DB_SF" -eq 0 ]; then
    status=FULL_LIFT
  elif [ "$DB_LG" -gt 0 ]; then
    status=PARTIAL_LIFT
  else
    status=NO_LIFT
  fi
  printf '%-6s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s   %s\n' \
    "$b" "$CG_LG" "$CG_AF" "$CG_SF" "$RS_LG" "$RS_AF" "$RS_SF" \
    "$DB_LG" "$DB_AF" "$DB_SF" "$status"
done
