// RUN: cgeist %s --function='*' -S | FileCheck %s

// `float op= double` is computed in double (the computation type) and the
// result is rounded to float once.

// CHECK-LABEL: func.func @add(
// CHECK: %[[F:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK: %[[S:.*]] = arith.addf %[[F]], %{{.*}} : f64
// CHECK: %[[R:.*]] = arith.truncf %[[S]] : f64 to f32
// CHECK: affine.store %[[R]]
void add(float *f, double d) { *f += d; }

// CHECK-LABEL: func.func @div(
// CHECK: %[[F:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK: %[[S:.*]] = arith.divf %[[F]], %{{.*}} : f64
// CHECK: %[[R:.*]] = arith.truncf %[[S]] : f64 to f32
// CHECK: affine.store %[[R]]
void div(float *f, double d) { *f /= d; }

// The computation type of `float += int` is float.
// CHECK-LABEL: func.func @add_int(
// CHECK: %[[I:.*]] = arith.sitofp %{{.*}} : i32 to f32
// CHECK: arith.addf %{{.*}}, %[[I]] : f32
void add_int(float *f, int i) { *f += i; }
