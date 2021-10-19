

// a = A[i]; A[i] *= a; A[i] += a;
func @foo(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    %1 = mulf %0, %0 : f32
    affine.store %1, %A[%i] : memref<10xf32>
    // Should replace %0 by a load from a scratchpad.
    %2 = addf %1, %0 : f32
    affine.store %2, %A[%i] : memref<10xf32>
  }
  return
}
