// --hls-pragma-report prints, for every HLS pragma, each op it ended up on
// after the whole pipeline (including inlined copies) or why it was not
// applied. Every pragma gets at least one line.
// RUN: cgeist %s --function='*' -S --hls-annotate --hls-pragma-report -O3 --raise-scf-to-affine -o /dev/null 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOREPORT

// NOREPORT-NOT: remark:

int gbuf[16];

void helper(int *a) {
  int local[8];
// CHECK-DAG: report.c:[[@LINE+2]]:1: remark: '#pragma HLS array_partition variable=local complete' -> memref.alloca (variable 'local') in @helper
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS array_partition variable=local complete' -> memref.alloca (variable 'local') in @kernel
#pragma HLS array_partition variable=local complete
  for (int i = 0; i < 8; ++i)
    local[i] = a[i];
  for (int i = 0; i < 8; ++i)
    a[i] = local[7 - i] + gbuf[i];
}

void kernel(int *a, int n) {
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS interface m_axi port=a' -> argument #0 (variable 'a') of @kernel
#pragma HLS interface m_axi port=a
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS inline off' -> func.func @kernel
#pragma HLS inline off
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS reset variable=gbuf' -> memref.global @gbuf
#pragma HLS reset variable=gbuf
  helper(a);
outer:
  for (int i = 0; i < n; ++i) {
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS pipeline II=1' -> affine.for (loop 'outer') in @kernel
#pragma HLS pipeline II=1
    a[i] += 1;
  }
  int dead = 0;
  for (int i = 0; i < n; ++i) {
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS unroll factor=2' -> not applied: the loop was removed during optimization
#pragma HLS unroll factor=2
    dead += i;
  }
  for (int i = 0; i < n; ++i) {
#pragma HLS loop_tripcount max=4
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS loop_tripcount max=4' -> ignored: duplicate of the pragma at {{.*}}report.c:[[@LINE-1]]:1
#pragma HLS loop_tripcount max=4
    a[i] -= 1;
  }
}

// Never emitted: static and unused.
static void unused(int *a) {
// CHECK-DAG: report.c:[[@LINE+1]]:1: remark: '#pragma HLS pipeline' -> not applied: the enclosing function is not emitted
#pragma HLS pipeline
  a[0] = 0;
}
