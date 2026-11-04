// `float op= double` computes in double and rounds the result to float once
// (C's computation type), not on a RHS rounded to float first.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine -o - | FileCheck %s

// CHECK-LABEL: func.func @add_d(
// CHECK: %[[F:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK: %[[S:.*]] = arith.addf %[[F]], %{{.*}} : f64
// CHECK: arith.truncf %[[S]] : f64 to f32
void add_d(float *f, double d) { *f += d; }

// CHECK-LABEL: func.func @sub_d(
// CHECK: %[[F:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK: %[[S:.*]] = arith.subf %[[F]], %{{.*}} : f64
// CHECK: arith.truncf %[[S]] : f64 to f32
void sub_d(float *f, double d) { *f -= d; }

// CHECK-LABEL: func.func @mul_d(
// CHECK: %[[F:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK: %[[S:.*]] = arith.mulf %[[F]], %{{.*}} : f64
// CHECK: arith.truncf %[[S]] : f64 to f32
void mul_d(float *f, double d) { *f *= d; }

// CHECK-LABEL: func.func @div_d(
// CHECK: %[[F:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK: %[[S:.*]] = arith.divf %[[F]], %{{.*}} : f64
// CHECK: arith.truncf %[[S]] : f64 to f32
void div_d(float *f, double d) { *f /= d; }

// float += float stays in float.
// CHECK-LABEL: func.func @add_f(
// CHECK-NOT: arith.extf
// CHECK: arith.addf %{{.*}} : f32
void add_f(float *f, float g) { *f += g; }

// A float accumulator of double products: summed in double, rounded each
// step.
// CHECK-LABEL: func.func @acc(
// CHECK: affine.for %{{.*}} = 0 to 16 iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK: %[[P:.*]] = arith.mulf %{{.*}} : f64
// CHECK: %[[A:.*]] = arith.extf %[[ACC]] : f32 to f64
// CHECK: %[[S:.*]] = arith.addf %[[A]], %[[P]] : f64
// CHECK: %[[R:.*]] = arith.truncf %[[S]] : f64 to f32
// CHECK: affine.yield %[[R]] : f32
void acc(float A[16], float *O) {
  float s = 0;
  for (int i = 0; i < 16; i++)
    s += 0.1 * A[i];
  *O = s;
}
