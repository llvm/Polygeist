// #pragma HLS bind_op: hls.bind_op = [{variable, op, impl?, latency?}] on
// each operation of kind `op` whose result is assigned to the variable (by
// its initializer, `=` or a compound assignment). f*/d*/h* ops match float
// operations of 32/64/16 bits.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

double sqrt(double);

// CHECK-LABEL: func.func @kinds(
// CHECK-DAG: arith.muli %arg0, %arg1 {hls.bind_op = [{impl = "dsp", latency = 3 : i32, op = "mul", variable = "c"}]} : i32
// CHECK-DAG: arith.addi %arg0, %{{.*}} {hls.bind_op = [{impl = "fabric", op = "add", variable = "s"}]} : i32
// CHECK-DAG: arith.addi %{{.*}}, %arg1 {hls.bind_op = [{impl = "fabric", op = "add", variable = "s"}]} : i32
// CHECK-DAG: arith.subi %arg0, %arg1 {hls.bind_op = [{op = "sub", variable = "d"}]} : i32
// CHECK-DAG: arith.mulf %arg2, %arg3 {hls.bind_op = [{impl = "maxdsp", op = "fmul", variable = "f"}]} : f32
// CHECK-DAG: arith.divf %arg2, %arg3 {hls.bind_op = [{impl = "fabric", latency = 8 : i32, op = "fdiv", variable = "g"}]} : f32
// CHECK-DAG: math.sqrt %arg4 {hls.bind_op = [{impl = "fabric", op = "dsqrt", variable = "q"}]} : f64
// CHECK-DAG: arith.addf %arg4, %arg4 {hls.bind_op = [{impl = "fulldsp", op = "dadd", variable = "r"}]} : f64
int kinds(int a, int b, float x, float y, double z) {
  int c = a * b;
#pragma HLS bind_op variable=c op=mul impl=dsp latency=3
  int s;
#pragma HLS BIND_OP variable=s op=ADD impl=fabric
  s = a + c;
  s += b;
  int d = a - b;
#pragma HLS bind_op variable=d op=sub
  float f = x * y;
#pragma HLS bind_op variable=f op=fmul impl=maxdsp
  float g = x / y;
#pragma HLS bind_op variable=g op=fdiv impl=fabric latency=8
  double q = sqrt(z);
#pragma HLS bind_op variable=q op=dsqrt impl=fabric
  double r = z + z;
#pragma HLS bind_op variable=r op=dadd impl=fulldsp
  return s + d + (int)f + (int)g + (int)q + (int)r;
}

// In a loop, the operation assigned in every iteration is annotated.
// CHECK-LABEL: func.func @in_loop(
// CHECK: arith.muli {{.*}} {hls.bind_op = [{impl = "dsp", op = "mul", variable = "prod"}]} : i32
void in_loop(int *a, int *b, int *c) {
  for (int i = 0; i < 16; ++i) {
    int prod;
#pragma HLS bind_op variable=prod op=mul impl=dsp
    prod = a[i] * b[i];
    c[i] = prod;
  }
}

// After inlining, the cloned operation keeps the attribute.
// CHECK-LABEL: func.func @top(
// NOINLINE: call @in_loop
// INLINE: arith.muli {{.*}} {hls.bind_op = [{impl = "dsp", op = "mul", variable = "prod"}]} : i32
void top(int *a, int *b, int *c) { in_loop(a, b, c); }

#ifdef ERRORS
int g_acc;
int errors(int a, int b, float x) {
  int m = a + b;
// ERR-DAG: bind_op.c:[[@LINE+1]]:1: warning: '#pragma HLS bind_op': no matching operation is assigned to variable 'm'; the pragma has no effect
#pragma HLS bind_op variable=m op=mul impl=dsp
  float f = x * x;
// A double op on a float variable does not match either.
// ERR-DAG: bind_op.c:[[@LINE+1]]:1: warning: '#pragma HLS bind_op': no matching operation is assigned to variable 'f'; the pragma has no effect
#pragma HLS bind_op variable=f op=dmul
// ERR-DAG: bind_op.c:[[@LINE+1]]:1: error: '#pragma HLS bind_op' requires option 'op'
#pragma HLS bind_op variable=m impl=dsp
// ERR-DAG: bind_op.c:[[@LINE+1]]:35: error: invalid value 'fma' for option 'op' in '#pragma HLS bind_op'
#pragma HLS bind_op variable=m op=fma
// ERR-DAG: bind_op.c:[[@LINE+1]]:21: error: '#pragma HLS bind_op' requires a local variable or parameter
#pragma HLS bind_op variable=g_acc op=add
  g_acc = m;
  return m + (int)f;
}
#endif
