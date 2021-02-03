#map0 = affine_map<()[s0] -> (s0 - 1)>

func @jacobi_1d(%A: memref<?xf32>, %B: memref<?xf32>, %T: index, %N: index) {
  affine.for %t = 0 to %T {
    affine.for %i = 2 to #map0()[%N] {
      %cst = constant 0.333333 : f32
      %0 = affine.load %A[%i - 1] : memref<?xf32>
      %1 = affine.load %A[%i] : memref<?xf32>
      %2 = affine.load %A[%i + 1] : memref<?xf32>
      %3 = addf %0, %1 : f32
      %4 = addf %2, %3 : f32
      %5 = mulf %cst, %4 : f32
      affine.store %5, %B[%i] : memref<?xf32>
    }

    affine.for %i = 2 to #map0()[%N] {
      %0 = affine.load %B[%i] : memref<?xf32>
      affine.store %0, %A[%i] : memref<?xf32>
    }
  }
  return
}
