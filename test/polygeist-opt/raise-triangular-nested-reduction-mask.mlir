// RUN: polygeist-opt %S/../../issues/polybench_section42/ir/correlation/orig.mlir \
// RUN:   --select-func=func-name=kernel_correlation --remove-iter-args \
// RUN:   --affine-parallelize --raise-affine-to-linalg-pipeline | FileCheck %s

// A dynamic triangular wrapper around a nested dot-product must gate the
// nested generic's yield.  Otherwise `for j = i + 1 .. M` becomes a full
// rectangle and adds the dot product to the diagonal and lower triangle.

// CHECK-LABEL: func.func @kernel_correlation
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: ^bb0({{.*}}%[[OLD:[A-Za-z0-9_]+]]: f64):
// CHECK: %[[ROW:.*]] = linalg.index 0 : index
// CHECK: %[[COL:.*]] = linalg.index 1 : index
// CHECK: %[[LB:.*]] = affine.apply {{.*}}(%[[ROW]])
// CHECK: %[[ACTIVE:.*]] = arith.cmpi sge, %[[COL]], %[[LB]] : index
// CHECK: %[[GATED:.*]] = arith.select %[[ACTIVE]], {{.*}}, %[[OLD]] : f64
// CHECK: linalg.yield %[[GATED]] : f64
