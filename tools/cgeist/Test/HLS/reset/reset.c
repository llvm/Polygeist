// #pragma HLS reset: hls.reset = [{variable}] on the memref.global of a
// global or static variable; `off` gives hls.reset_off = [{variable}].
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

int history[8];
int counter;

// CHECK-DAG: memref.global @history : memref<8xi32> = uninitialized {hls.reset = [{variable = "history"}]}
// CHECK-DAG: memref.global @counter : memref<1xi32> = uninitialized {hls.reset_off = [{variable = "counter"}]}
// CHECK-DAG: memref.global "private" @"accumulate@static@state{{.*}}" : memref<4xi32> = {{.*}} {hls.reset = [{variable = "state"}]}
int accumulate(int x) {
  static int state[4] = {0, 0, 0, 0};
#pragma HLS reset variable=history
#pragma HLS RESET variable=counter off
#pragma HLS reset variable=state
  for (int i = 7; i > 0; --i)
    history[i] = history[i - 1];
  history[0] = x;
  state[x & 3] += x;
  counter += 1;
  return history[7] + state[0] + counter;
}

#ifdef ERRORS
void errors(int *a, int p) {
  int local = 0;
// ERR: reset.c:[[@LINE+1]]:28: error: '#pragma HLS reset' requires a static or global variable, but 'local' is not
#pragma HLS reset variable=local
// ERR: reset.c:[[@LINE+1]]:28: error: '#pragma HLS reset' requires a static or global variable, but 'p' is not
#pragma HLS reset variable=p
  a[0] = local + p;
}
#endif
