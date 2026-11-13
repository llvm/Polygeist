#!/bin/bash
# Gate (d): production paper pipeline on PolyBench, read-only.
#   run_kernel_e2e.sh <dir> <k> --debuf   (campaign recipe; PASS/FAIL)
#   kernel_match_rewrite.py --enable-structured-rewrite std.mlir (the paper's
#   matcher sweep, issues/polybench_section42/run_matcher_sweep.sh) -> launch set
# usage: gate_d.sh old|new OUT_DIR [kernel...]
# "new" runs the unmodified scripts against build-v2/bin through a shadow
# POLYGEIST_ROOT whose build/bin -> build-v2/bin (common_env.sh falls back to
# PATH=$POLYGEIST_ROOT/build/bin when that root has no envsetup.sh).
WHICH=$1; OUT=$(realpath -m $2); shift 2
ROOT=/home/arjaiswal/Polygeist
mkdir -p $OUT
if [ "$WHICH" = new ]; then
  SH=${SHADOW:-$ROOT/issues/raise_fixes/shadow_root_v2}
  mkdir -p $SH/build
  ln -sfn ${NEWBIN:-$ROOT/build-v2/bin} $SH/build/bin
  for d in llvm-project tools scripts generic_solver runtime include lib issues; do ln -sfn $ROOT/$d $SH/$d; done
  export POLYGEIST_ROOT=$SH
fi
KS=("$@")
if [ ${#KS[@]} = 0 ]; then
  KS=($(tail -n +2 $ROOT/issues/polybench_section42/manifest.csv | cut -d, -f1))
fi
run_one() {
  k=$1; src=$(grep "^$k," $ROOT/issues/polybench_section42/manifest.csv | cut -d, -f3)
  dir=$ROOT/$(dirname $src)
  o=$OUT/$k; mkdir -p $o
  POLYBENCH_E2E_OUTPUT_ROOT=$o timeout 900 bash $ROOT/scripts/correctness/run_kernel_e2e.sh $dir $k --debuf > $o/e2e.log 2>&1
  rc=$?
  std=$o/e2e_${k}_debuf/std.mlir
  timeout 300 /usr/bin/python3 $ROOT/scripts/correctness/kernel_match_rewrite.py --enable-structured-rewrite $std > $o/matched.mlir 2> $o/matcher.log
  mrc=$?
  n=$(grep -cE '\bkernel\.launch ' $o/matched.mlir 2>/dev/null || true)
  syms=$(sed -n 's/.*kernel\.launch @\([^ (]*\).*/\1/p' $o/matched.mlir | sort | paste -sd+ -)
  echo "$k e2e_rc=$rc result=$(tail -1 $o/e2e.log | cut -c1-60 | tr ' ' '_') match_rc=$mrc launches=${n:-0} syms=$syms"
}
export -f run_one; export OUT ROOT
printf "%s\n" "${KS[@]}" | xargs -P 8 -I{} bash -c 'run_one {}' | sort > $OUT/summary.txt
cat $OUT/summary.txt
