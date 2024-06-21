// RUN: if [ %polymer_enabled == 1 ]; then polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=tadashi -debug-only=tadashi-opt,islscop %s; fi

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[] : (-d0 + 50 -1 >= 0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?xf32>,
            %A: memref<?xf32>,
            %B: memref<?xf32>,
            %S: index,
            %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    affine.for %i = 0 to 100 {
      affine.if #set1(%i)[] {
        %4 = affine.load %B[%i] : memref<?xf32>
        affine.store %4, %C[%i] : memref<?xf32>
      } else {
        %4 = affine.load %A[%i] : memref<?xf32>
        affine.store %4, %C[%i] : memref<?xf32>
      }
      affine.store %beta, %C[%i] : memref<?xf32>
    }
    return
  }
}
