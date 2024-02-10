func @main() {
  %A = alloc() : memref<64x64xf32>
  %n = constant 64 : i32
  call @kernel(%A, %n): (memref<64x64xf32>, i32) -> ()
  return
}

func @kernel(%A : memref<64x64xf32>, %n : i32) {
  %N = index_cast %n : i32 to index

  affine.for %i = 1 to %N {
    affine.for %j = 1 to %N {
      %0 = affine.load %A[%i - 1, %j] : memref<64x64xf32>
      %1 = affine.load %A[%i, %j - 1] : memref<64x64xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %A[%i, %j] : memref<64x64xf32>
    }
  }

  return
}
