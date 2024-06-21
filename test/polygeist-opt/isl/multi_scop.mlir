// RUN: if [ %polymer_enabled == 1 ]; then polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=tadashi -debug-only=tadashi-opt,islscop %s; fi

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
module {
  func.func private @foo()
  func.func @f1(%beta: f32,
            %C: memref<?xf32>,
            %N: index) {
    affine.for %i = 0 to #map()[%N] step 1 {
      affine.store %beta, %C[%i + 1] : memref<?xf32>
    }
    func.call @foo() : () -> ()
    // TODO these should be the same scop
    affine.for %i = 0 to #map()[%N] step 2 {
      affine.store %beta, %C[%i + 2] : memref<?xf32>
    }
    affine.for %i = 0 to #map()[%N] step 3 {
      affine.store %beta, %C[%i + 3] : memref<?xf32>
    }
    return
  }
  func.func @f2(%beta: f32,
            %C: memref<?xf32>,
            %N: index) {
    affine.for %i = 0 to #map()[%N] step 4{
      affine.store %beta, %C[%i + 4] : memref<?xf32>
    }
    return
  }
}
