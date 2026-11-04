T=/mnt/d/HLS/Polygeist/tools/cgeist/Test/polybench/linear-algebra/solvers/trisolv/trisolv.c
I=/mnt/d/HLS/Polygeist/tools/cgeist/Test/polybench/utilities
R=/mnt/d/HLS/Polygeist/llvm-project/build/lib/clang/18

# bare global cl::opt — no -mllvm wrapper
/mnt/d/HLS/Polygeist/build/bin/cgeist "$T" -resource-dir="$R" -I "$I" -O3 \
  --print-before-all -o /tmp/t.execm 2>/tmp/cg.log
echo "exit: $? lines: $(wc -l < /tmp/cg.log)"
grep -iE "Dump (Before|After)" /tmp/cg.log | tail -6