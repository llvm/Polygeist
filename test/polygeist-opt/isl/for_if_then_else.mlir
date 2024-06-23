// RUN: if [ %polymer_enabled == 1 ]; then polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal $ISL_OPT_PLACEHOLDER %s; fi 2>1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?xf32>,
            %A: memref<?xf32>,
            %B: memref<?xf32>,
            %S: index,
            %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    affine.for %i = 0 to #map()[%N] {
      affine.if #set1(%i)[%S] {
        %4 = affine.load %B[%i] : memref<?xf32>
        affine.store %4, %C[%i] : memref<?xf32>
      } else {
        %4 = affine.load %A[%i] : memref<?xf32>
        affine.store %4, %C[%i] : memref<?xf32>
      }
    }
    return
  }
}
