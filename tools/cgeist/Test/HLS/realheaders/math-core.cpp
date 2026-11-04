// hls:: math functions of the real hls_math.h are cores of the Vitis math
// library (XlxMathCore.h): each call carries hls.math_core with the
// function and, for a fixed-point overload, the argument's ap_fixed<W,I> /
// ap_ufixed<W,I>, which hls-est's math-cost model prices per instance
// (MathCores.cpp, VITIS_NOTES N16) as it prices the stand-in headers'
// __hlsest_math_<fn> markers. The callee keeps its body (the headers' values)
// and is never inlined; std:: functions and non-core hls:: functions are not
// marked.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: FileCheck %s --check-prefix=DEF < %t.mlir
#include <ap_fixed.h>
#include <hls_math.h>

// CHECK-LABEL: func.func @fx(
// CHECK-DAG: call @_ZN3hls4tanhILi16ELi5EEE{{.*}} {hls.math_core = {function = "tanh", type = "ap_fixed<16,5>"}} : (i16) -> i16
// CHECK-DAG: call @_ZN3hls4sqrtILi12ELi4EEE{{.*}} {hls.math_core = {function = "sqrt", type = "ap_ufixed<12,4>"}} : (i12) -> i12
// CHECK-DAG: call @_ZN3hls3expILi32ELi16EEE{{.*}} {hls.math_core = {function = "exp", type = "ap_fixed<32,16>"}} : (i32) -> i32
// CHECK: return
extern "C" void fx(ap_fixed<16, 5> a, ap_ufixed<12, 4> b, ap_fixed<32, 16> c,
                   ap_fixed<16, 5> *o1, ap_ufixed<12, 4> *o2,
                   ap_fixed<32, 16> *o3) {
  *o1 = hls::tanh(a);
  *o2 = hls::sqrt(b);
  *o3 = hls::exp(c);
}

// The float / double / half overloads are declarations (the closed library):
// external calls, marked with the function only (the type is the operand's).
// Those cgeist already lowers by name to a math op (hls::pow -> math.powf,
// hls::log -> math.log, as before) are cores to hls-est as math ops.
// CHECK-LABEL: func.func @fl(
// CHECK-DAG: math.powf {{.*}} : f32
// CHECK-DAG: call @_ZN3hls4sinfEf({{.*}}) {hls.math_core = {function = "sin"}} : (f32) -> f32
// CHECK-DAG: math.log {{.*}} : f64
// CHECK-DAG: math.absf
// CHECK: return
extern "C" void fl(float x, double y, float *o1, float *o2, double *o3,
                   double *o4) {
  *o1 = hls::pow(x, x);
  *o2 = hls::sinf(x);
  *o3 = hls::log(y);
  *o4 = std::fabs(y);
}

// The kept callees: bodies (fixed point), declarations (float).
// DEF-DAG: func.func @_ZN3hls4tanhILi16ELi5EEE{{.*}}(%arg0: i16) -> i16 attributes {hls.math_core = {function = "tanh", type = "ap_fixed<16,5>"}
// DEF-DAG: func.func private @_ZN3hls4sinfEf(f32) -> f32 attributes {hls.math_core = {function = "sin"}
