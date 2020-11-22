#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @gemm(%arg0: f32, %arg1: f32, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg2, %c0 : memref<?x?xf32>
    %1 = dim %arg2, %c1 : memref<?x?xf32>
    %2 = dim %arg3, %c1 : memref<?x?xf32>
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to %1 {
        call @S0(%arg2, %arg5, %arg6, %arg1) : (memref<?x?xf32>, index, index, f32) -> ()
      }
      affine.for %arg6 = 0 to %2 {
        affine.for %arg7 = 0 to %1 {
          call @S1(%arg2, %arg5, %arg7, %arg4, %arg6, %arg0, %arg3) : (memref<?x?xf32>, index, index, memref<?x?xf32>, index, f32, memref<?x?xf32>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: f32) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<?x?xf32>
    %1 = mulf %0, %arg3 : f32
    affine.store %1, %arg0[%arg1, %arg2] : memref<?x?xf32>
    return
  }
  func @S1(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: memref<?x?xf32>, %arg4: index, %arg5: f32, %arg6: memref<?x?xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<?x?xf32>
    %1 = affine.load %arg6[%arg1, %arg4] : memref<?x?xf32>
    %2 = mulf %arg5, %1 : f32
    %3 = affine.load %arg3[%arg4, %arg2] : memref<?x?xf32>
    %4 = mulf %2, %3 : f32
    %5 = addf %0, %4 : f32
    affine.store %5, %arg0[%arg1, %arg2] : memref<?x?xf32>
    return
  }
}
