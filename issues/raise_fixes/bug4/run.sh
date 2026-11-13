#!/bin/bash
# usage: run.sh BIN_DIR   -- campaign recipe of run_kernel_e2e.sh --debuf
B=$1; cd $(dirname $0)
$B/cgeist gemv.c --function=gemv --resource-dir=/usr/lib/clang/14 -O1 --raise-scf-to-affine -S -o gemv.mlir
$B/polygeist-opt --select-func=func-name=gemv --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline --lower-polygeist-submap gemv.mlir -o gemv_raised.mlir
for o in --linalg-debufferize --linalg-debufferize=use-multi-root=true; do
  start=$(date +%s.%N); timeout 30 $B/polygeist-opt $o gemv_raised.mlir -o /dev/null; rc=$?
  echo "$o rc=$rc ($(echo "$(date +%s.%N) - $start" | bc | cut -c1-5)s)"; done
