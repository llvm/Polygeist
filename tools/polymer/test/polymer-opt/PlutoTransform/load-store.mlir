func @load_store() -> () {
  %A = alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %0 = affine.load %A[%i] : memref<64xf32>
    affine.store %0, %A[%i] : memref<64xf32>
  }
  return
}
