// RUN: if [ %polymer_enabled == 1 ]; then polygeist-opt --polyhedral-opt %s 2>&1 | FileCheck %s; fi

// CHECK: <OpenScop>
// CHECK: </OpenScop>
// CHECK-NOT: __polygeist_outlined
// CHECK-NOT: S[0-9]+
#map = affine_map<()[s0] -> (s0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?x?xf32>,
            %A: memref<?x?xf32>,
            %B: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %NI = memref.dim %C, %c0 : memref<?x?xf32>
    %NJ = memref.dim %C, %c1 : memref<?x?xf32>
    %NK = memref.dim %A, %c1 : memref<?x?xf32>

    affine.for %i = 0 to #map()[%NI] {
      affine.for %j = 0 to #map()[%NJ] {
        %0 = affine.load %C[%i, %j] : memref<?x?xf32>
        %1 = arith.mulf %0, %beta : f32
        affine.store %1, %C[%i, %j] : memref<?x?xf32>
      }

      affine.for %j = 0 to #map()[%NJ] {
        affine.for %k = 0 to #map()[%NK] {
          %2 = affine.load %A[%i, %k] : memref<?x?xf32>
          %3 = arith.mulf %alpha, %2 : f32
          %4 = affine.load %B[%k, %j] : memref<?x?xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = affine.load %C[%i, %j] : memref<?x?xf32>
          %7 = arith.addf %6, %5 : f32
          affine.store %7, %C[%i, %j] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
