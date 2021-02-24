func @simple(%A: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %N = dim %A, %c0: memref<?x?xf32>
  %M = dim %A, %c1: memref<?x?xf32>
  affine.for %i = 1 to %N {
    affine.for %j = 1 to %M {
      %0 = affine.load %A[%i - 1, %j] : memref<?x?xf32>
      %1 = affine.load %A[%i, %j - 1] : memref<?x?xf32>
      %2 = addf %0, %1: f32
      affine.store %2, %A[%i, %j]: memref<?x?xf32>
    }
  } 
  return
}
