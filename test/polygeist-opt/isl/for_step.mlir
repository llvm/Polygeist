// RUN: if [ %polymer_enabled == 1 ]; then polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=tadashi -debug-only=tadashi-opt,islscop %s; fi

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?xf32>,
            %A: memref<?xf32>,
            %B: memref<?xf32>,
            %S: index,
            %N: index) {
    affine.for %i = 0 to #map()[%N] step 2 {
      affine.store %beta, %C[%i] : memref<?xf32>
    }
    return
  }
}
