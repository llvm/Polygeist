// Load data from one 1D array to another 2D array.
// This case is used to test the case that two statements that
// have not completely identical parameter sets.

func @load_store_param_2d(%A : memref<?xf32>, %B : memref<?x?xf32>) -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %M = dim %B, %c0 : memref<?x?xf32>
  %N = dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %M {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.for %j = 0 to %N {
      affine.store %0, %B[%i, %j] : memref<?x?xf32>
    }
  }

  return 
}

// TODO: fix this test
