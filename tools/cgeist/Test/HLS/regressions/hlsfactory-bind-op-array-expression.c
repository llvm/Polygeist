// Regression (HLSFactory FLAT-acceleration systolic array):
//   local_out[t][h] = query * key[t][h] + prev;
//   #pragma HLS BIND_OP variable=local_out op=mul impl=dsp
// was not applied: only plain-variable assignments whose top operation was
// the bound one were considered. Array-element and member assignments now
// count, and the expression assigned is searched (without crossing loads).
// RUN: cgeist %s --function='*' -S --hls-annotate 2>/dev/null | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

// WARN-NOT: 'local_out'
// WARN: warning: '#pragma HLS bind_op': no matching operation is assigned to variable 'd'
// CHECK-LABEL: func.func @systolic(
// CHECK: arith.mulf {{.*}} {hls.bind_op = [{impl = "dsp", latency = -1 : i32, op = "fmul", variable = "local_out"}]} : f32
// CHECK-NOT: hls.bind_op
// CHECK: return
void systolic(float key[4][4], float *q, float out[4][4]) {
  float local_out[4][4];
  for (int t = 0; t < 4; ++t)
    for (int h = 0; h < 4; ++h) {
      float prev = h == 0 ? 0.0f : local_out[t][h - 1];
#pragma HLS BIND_OP variable=local_out op=fmul impl=dsp latency=-1
      local_out[t][h] = q[t] * key[t][h] + prev;
    }
  for (int t = 0; t < 4; ++t)
    for (int h = 0; h < 4; ++h)
      out[t][h] = local_out[t][h];
}

// A previous statement's multiplication is not reached through the load of
// `c`: only this assignment's own expression is searched.
// CHECK-LABEL: func.func @separate_statements(
// CHECK: arith.muli %arg0, %arg1 : i32
// CHECK-NOT: hls.bind_op
// CHECK: return
int separate_statements(int a, int b, int e) {
  volatile int c = a * b;
  int d;
#pragma HLS bind_op variable=d op=mul impl=dsp
  d = c + e;
  return d;
}
