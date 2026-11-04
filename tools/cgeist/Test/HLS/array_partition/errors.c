// Semantic errors and warnings for array_partition. Each produces a specific
// diagnostic and a clean exit.
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

int scalar_global;

void f(int *a, int n) {
  int buf[16];
  int m[4][8];
  int scalar = 0;
// CHECK-DAG: errors.c:[[@LINE+1]]:1: error: '#pragma HLS array_partition' with type=cyclic requires option 'factor'
#pragma HLS array_partition variable=buf cyclic
// CHECK-DAG: errors.c:[[@LINE+1]]:1: error: '#pragma HLS array_partition' with type=block requires option 'factor'
#pragma HLS array_partition variable=buf type=block
// Ignored with a warning, as Vitis 2025.1 ignores it ([HLS 214-253]).
// CHECK-DAG: errors.c:[[@LINE+1]]:40: warning: dim=3 is out of range in '#pragma HLS array_partition': 'm' has 2 dimension(s); the pragma is ignored, as Vitis ignores it
#pragma HLS array_partition variable=m dim=3 complete
// CHECK-DAG: errors.c:[[@LINE+1]]:29: error: '#pragma HLS array_partition' requires an array or pointer, but 'scalar' has type 'int'
#pragma HLS array_partition variable=scalar complete
// CHECK-DAG: errors.c:[[@LINE+1]]:29: error: '#pragma HLS array_partition' requires an array or pointer, but 'n' has type 'int'
#pragma HLS array_partition variable=n complete
// CHECK-DAG: errors.c:[[@LINE+1]]:29: error: '#pragma HLS array_partition' requires an array or pointer, but 'scalar_global' has type 'int'
#pragma HLS array_partition variable=scalar_global complete
// CHECK-DAG: errors.c:[[@LINE+1]]:38: error: use of undeclared variable 'nope' in '#pragma HLS array_partition'
#pragma HLS array_partition variable=nope complete
// CHECK-DAG: errors.c:[[@LINE+1]]:42: error: no member named 'x' in 'int[16]' (in '#pragma HLS array_partition')
#pragma HLS array_partition variable=buf.x complete
// CHECK-DAG: errors.c:[[@LINE+1]]:55: warning: 'factor' is ignored for complete partitioning in '#pragma HLS array_partition'
#pragma HLS array_partition variable=m complete dim=1 factor=2
// CHECK-DAG: errors.c:[[@LINE+1]]:47: warning: factor=32 exceeds the size 8 of dimension 2 of 'm'
#pragma HLS array_partition variable=m cyclic factor=32 dim=2
  for (int i = 0; i < 16; ++i)
    buf[i] = a[i] + m[i % 4][i % 8] + scalar + n + scalar_global;
  a[0] = buf[3];
}

// Two pragmas on one dimension (dim=0 covers every dimension): cyclic against
// block and `off` against a partition are errors, an identical repeat is a
// warning; the others are merged the Vitis way (W10,
// regressions/array-partition-merge.c): here `complete dim=0` replaces the
// cyclic partition of dimension 1, with a remark at the dropped pragma.
void conflicts(int *a) {
  int buf[4][4];
// CHECK-DAG: errors.c:[[@LINE+1]]:1: remark: '#pragma HLS array_partition' (cyclic factor=2) on dimension 1 of variable 'buf' is dropped: the later pragma (complete) partitions it more finely, which Vitis keeps
#pragma HLS array_partition variable=buf cyclic factor=2 dim=1
// CHECK-DAG: errors.c:[[@LINE+1]]:1: error: '#pragma HLS array_partition' conflicts with an earlier pragma on the same variable 'buf'
#pragma HLS array_partition variable=buf block factor=2 dim=1
// CHECK-DAG: errors.c:[[@LINE+1]]:1: warning: duplicate '#pragma HLS array_partition' on the same variable 'buf' is ignored
#pragma HLS array_partition variable=buf type=cyclic factor=2 dim=1
#pragma HLS array_partition variable=buf complete dim=0
// CHECK-DAG: errors.c:[[@LINE+1]]:1: error: '#pragma HLS array_partition' conflicts with an earlier pragma on the same variable 'buf'
#pragma HLS array_partition variable=buf off
  for (int i = 0; i < 16; ++i)
    buf[i / 4][i % 4] = a[i];
  a[0] = buf[1][2];
}

// CHECK-NOT: Stack dump
