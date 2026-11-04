// std::pow(float, int) is libstdc++'s promoted template: parameters (float,
// int), result double (C++11 [c.math]), computed as pow((double)x, n). The
// float argument is converted to the result type before the op, so the op's
// operand and result types agree (CIRCT W9r14 review, kernel
// w9r14rv_pw_f32: Vitis builds pow_generic<double>; before: llvm.intr.powi
// (f32, i32) -> f64, computed in float).
// RUN: cgeist %s %resourcedir --function='*' -S -o - | FileCheck %s
#include <cmath>

// CHECK-LABEL: func.func @_Z7squaredf(
// CHECK: %[[X:.+]] = arith.extf %{{.*}} : f32 to f64
// CHECK: llvm.intr.powi(%[[X]], %{{.*}}) : (f64, i32) -> f64
double squared(float x) { return std::pow(x, 2); }

// CHECK-LABEL: func.func @_Z14squared_doubled(
// CHECK-NOT: arith.extf
// CHECK: llvm.intr.powi(%{{.*}}, %{{.*}}) : (f64, i32) -> f64
double squared_double(double x) { return std::pow(x, 2); }
