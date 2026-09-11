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
}

// CHECK-LABEL: func.func @inclusive_scan
// CHECK: affine.for
// CHECK: affine.load {{.*}}[%{{.*}} - 1]
// CHECK: affine.store
// CHECK-NOT: linalg.generic
