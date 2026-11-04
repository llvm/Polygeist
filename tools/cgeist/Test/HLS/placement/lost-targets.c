// When the optimizer removes a pragma's target (a dead loop, a scalar
// promoted to a register, a folded operation), cgeist says so instead of
// dropping the pragma silently or attaching it elsewhere.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

void dead_loop(int n) {
  int j = 0;
  for (int i = 0; i < n; ++i) {
    // j is never used: the whole loop is dead.
// CHECK-DAG: lost-targets.c:[[@LINE+1]]:1: warning: '#pragma HLS pipeline': the loop it applies to was removed during optimization (e.g. fully unrolled or proven dead); the pragma has no effect
#pragma HLS pipeline II=1
    j += i;
  }
}

// Scalars are not preserved as memories; mem2reg removes their storage.
int scalar_var(int *a) {
  int scalar = a[0];
// CHECK-DAG: lost-targets.c:[[@LINE+1]]:1: warning: '#pragma HLS stream': variable 'scalar' was promoted to registers or removed during optimization; the pragma has no effect
#pragma HLS stream variable=scalar
  return scalar * 2;
}

// The multiplication is folded away (x * 1).
int folded(int x) {
  int one = 1;
  int m = x * one;
// CHECK-DAG: lost-targets.c:[[@LINE+1]]:1: warning: '#pragma HLS bind_op': the operation it binds was folded away during optimization; the pragma has no effect
#pragma HLS bind_op variable=m op=mul impl=dsp
  return m;
}

// The attributes are not moved to some other op.
// CHECK-NOT: hls.pipeline
// CHECK-NOT: hls.stream
// CHECK-NOT: hls.bind_op
