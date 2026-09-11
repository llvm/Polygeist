// RUN: polygeist-opt %s --raise-scf-to-affine | FileCheck %s

#nonempty = affine_set<()[s0] : (s0 - 1 >= 0)>
module {
  memref.global @size : memref<1xi32>
  func.func @guarded_bound(%guard: index, %out: memref<?xf64>) {
    %size = memref.get_global @size : memref<1xi32>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %value = arith.constant 0.0 : f64
    affine.if #nonempty()[%guard] {
      %n32 = affine.load %size[0] : memref<1xi32>
      %n = arith.index_cast %n32 : i32 to index
      scf.for %i = %zero to %n step %one {
        memref.store %value, %out[%i] : memref<?xf64>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @guarded_bound
// CHECK: %[[N32:.*]] = affine.load %{{.*}}[0]
// CHECK: %[[N:.*]] = arith.index_cast %[[N32]]
// CHECK: affine.if
// CHECK: affine.for %{{.*}} = 0 to %[[N]]
// CHECK-NOT: scf.for
