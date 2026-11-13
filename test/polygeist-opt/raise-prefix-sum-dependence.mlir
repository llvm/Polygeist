// RUN: polygeist-opt --raise-affine-to-linalg %s | FileCheck %s

module {
  func.func @inclusive_scan(%histogram: memref<256xi32>,
                            %cdf: memref<256xi32>) {
    %first = affine.load %histogram[0] : memref<256xi32>
    affine.store %first, %cdf[0] : memref<256xi32>
    affine.for %i = 1 to 256 {
      %previous = affine.load %cdf[%i - 1] : memref<256xi32>
      %value = affine.load %histogram[%i] : memref<256xi32>
      %next = arith.addi %previous, %value : i32
      affine.store %next, %cdf[%i] : memref<256xi32>
    }
    return
  }

// CHECK-LABEL: func.func @inclusive_scan
// CHECK: affine.for
// CHECK: affine.load {{.*}}[%{{.*}} - 1]
// CHECK: affine.store
// CHECK-NOT: linalg.generic

// A reduction may read a disjoint row prefix in the same allocation as its
// scalar destination. The upper bound proves that the prefix excludes it.
func.func @disjoint_row_reduction(%a: memref<32x32xf64>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to affine_map<(d0) -> (d0)>(%i) {
      affine.for %k = 0 to affine_map<(d0) -> (d0)>(%j) {
        %x = affine.load %a[%i, %k] : memref<32x32xf64>
        %y = affine.load %a[%j, %k] : memref<32x32xf64>
        %old = affine.load %a[%i, %j] : memref<32x32xf64>
        %product = arith.mulf %x, %y : f64
        %next = arith.subf %old, %product : f64
        affine.store %next, %a[%i, %j] : memref<32x32xf64>
      }
    }
  }
  return
}
// CHECK-LABEL: func.func @disjoint_row_reduction
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction"]
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: linalg.yield

}
