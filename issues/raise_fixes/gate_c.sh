#!/bin/bash
# Gate (c): BLAS corpus (blas/*.c, 21 functions) + debufferize stress corpus
# (debufferize_stress/inputs/*.mlir) through a given tool dir.
# usage: gate_c.sh BIN_DIR OUT_DIR
BIN=$1; OUT=$2
ROOT=/home/arjaiswal/Polygeist
mkdir -p "$OUT/blas" "$OUT/stress"
LIB=$ROOT/generic_solver/kernel_library.mlir
summary=$OUT/summary.txt; : > "$summary"
for pair in daxpy:daxpy daxpy:simple_daxpy daxpy:saxpy ddot:ddot ddot:simple_ddot ddot:sdot \
            dasum:dasum dasum:simple_dasum dasum:sasum dnrm2:dnrm2 dnrm2:simple_dnrm2 dnrm2:snrm2 \
            dscal:dscal dscal:simple_dscal dscal:sscal dcopy:dcopy dcopy:simple_dcopy dcopy:scopy \
            dgemm:dgemm dgemm:simple_dgemm dgemm:sgemm; do
  f=${pair%%:*}; fn=${pair##*:}; o=$OUT/blas/${f}_${fn}
  timeout 120 $BIN/cgeist $ROOT/blas/$f.c --function=$fn --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -fPIC -S -c -o $o.mlir 2>$o.err; r1=$?
  timeout 120 $BIN/polygeist-opt --select-func="func-name=$fn" --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline $o.mlir -o ${o}_linalg.mlir 2>>$o.err; r2=$?
  timeout 120 $BIN/polygeist-opt --linalg-debufferize ${o}_linalg.mlir -o ${o}_debuf.mlir 2>>$o.err; r3=$?
  timeout 120 $BIN/polygeist-opt --linalg-debufferize=use-multi-root=true ${o}_linalg.mlir -o ${o}_debufmr.mlir 2>>$o.err; r4=$?
  timeout 120 $BIN/polygeist-opt ${o}_debuf.mlir --linalg-to-kernel="kernel-library-path=$LIB" -o ${o}_kernel.mlir 2>>$o.err; r5=$?
  g=$(grep -c "linalg.generic" ${o}_linalg.mlir 2>/dev/null); lp=$(grep -cE "affine.for|scf.for" ${o}_linalg.mlir 2>/dev/null)
  echo "blas $f:$fn cgeist=$r1 raise=$r2 debuf=$r3 debufmr=$r4 kernel=$r5 generics=$g loops=$lp" >> "$summary"
done
for f in $ROOT/debufferize_stress/inputs/*.mlir; do
  b=$(basename $f .mlir); o=$OUT/stress/$b
  timeout 60 $BIN/polygeist-opt --linalg-debufferize $f -o $o.v2.mlir 2>$o.v2.err; r1=$?
  timeout 60 $BIN/polygeist-opt --linalg-debufferize=use-multi-root=true $f -o $o.mr.mlir 2>$o.mr.err; r2=$?
  echo "stress $b v2=$r1 mr=$r2" >> "$summary"
done
cat "$summary"
