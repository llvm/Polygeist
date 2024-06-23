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
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        affine.store %beta, %C[0] : memref<?xf32>
      }
    }
    return
  }
}
