// RUN: polymer-translate %s -export-scop | polymer-translate -import-scop | FileCheck %s

func @matmul() {
  %A = alloc() : memref<64x64xf32>
  %B = alloc() : memref<64x64xf32>
  %C = alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }

  return
}

// CHECK:      func @main(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to 64 {
// CHECK-NEXT:     affine.for %[[ARG4:.*]] = 0 to 64 {
// CHECK-NEXT:       affine.for %[[ARG5:.*]] = 0 to 64 {
// CHECK-NEXT:         call @S0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG0]], %[[ARG3]], %[[ARG4]], %[[ARG5]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK:      func private @S0(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index)
