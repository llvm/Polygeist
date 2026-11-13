// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @gather
  // CHECK-NOT: affine.for
  // CHECK: linalg.generic
  // CHECK: linalg.index 0
  // CHECK: linalg.index 1
  // CHECK: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x16xi32>
  // CHECK: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x32xf32>
  // CHECK: linalg.yield
  func.func @gather(%input: memref<8x32xf32>,
                    %indices: memref<8x16xi32>,
                    %output: memref<8x16xf32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 16 {
        %selected = affine.load %indices[%i, %j] : memref<8x16xi32>
        %k = arith.index_cast %selected : i32 to index
        %value = memref.load %input[%i, %k] : memref<8x32xf32>
        affine.store %value, %output[%i, %j] : memref<8x16xf32>
      }
    }
    return
  }
}
