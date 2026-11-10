#!/bin/bash
# compare two gate_c output dirs ignoring the nondeterministic dlti attribute order and paths
A=$1; B=$2; n=0; d=0
for f in $A/blas/*.mlir $A/stress/*.mlir; do g=${f/$A/$B}; n=$((n+1))
  if ! diff -q <(grep -v dlti $f | sed "s#$A#X#g") <(grep -v dlti $g | sed "s#$B#X#g") >/dev/null; then echo "DIFF $f"; d=$((d+1)); fi
done; echo "compared $n files, $d differ"
diff $A/summary.txt $B/summary.txt && echo "summaries identical"
