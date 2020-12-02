#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<()[s0, s1] -> (s0, s1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map6 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32)>
#map7 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map8 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map9 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map10 = affine_map<()[s0, s1] -> ((s0 + s1 - 2) floordiv 32 + 1)>


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
      affine.for %arg6 = 0 to %1 {
        affine.for %arg7 = 0 to %2 {
          call @S1(%arg2, %arg5, %arg6, %arg4, %arg7, %arg0, %arg3) : (memref<?x?xf32>, index, index, memref<?x?xf32>, index, f32, memref<?x?xf32>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: f32) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x?xf32>
    %1 = mulf %0, %arg3 : f32
    affine.store %1, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x?xf32>
    return
  }
  func @S1(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: memref<?x?xf32>, %arg4: index, %arg5: f32, %arg6: memref<?x?xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x?xf32>
    %1 = affine.load %arg6[symbol(%arg1), symbol(%arg4)] : memref<?x?xf32>
    %2 = mulf %arg5, %1 : f32
    %3 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x?xf32>
    %4 = mulf %2, %3 : f32
    %5 = addf %0, %4 : f32
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x?xf32>
    return
  }
  func @gemm_new(%arg0: f32, %arg1: f32, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = dim %arg3, %c1 : memref<?x?xf32>
    %1 = dim %arg2, %c1 : memref<?x?xf32>
    %2 = dim %arg2, %c0 : memref<?x?xf32>
    affine.for %arg5 = 0 to #map5()[%2] {
      affine.for %arg6 = 0 to #map5()[%1] {
        affine.for %arg7 = #map3(%arg5) to min #map4(%arg5)[%2] {
          affine.for %arg8 = #map3(%arg6) to min #map4(%arg6)[%1] {
            call @S0(%arg2, %arg7, %arg8, %arg1) : (memref<?x?xf32>, index, index, f32) -> ()
          }
        }
      }
    }
    affine.for %arg5 = 0 to #map10()[%1, %0] {
      affine.for %arg6 = max #map8(%arg5)[%0] to min #map9(%arg5)[%1] {
        affine.for %arg7 = 0 to #map5()[%2] {
          affine.for %arg8 = #map3(%arg6) to min #map4(%arg6)[%1] {
            affine.for %arg9 = #map3(%arg7) to min #map4(%arg7)[%2] {
              affine.for %arg10 = #map6(%arg5, %arg6) to min #map7(%arg5, %arg6)[%0] {
                call @S1(%arg2, %arg9, %arg8, %arg4, %arg10, %arg0, %arg3) : (memref<?x?xf32>, index, index, memref<?x?xf32>, index, f32, memref<?x?xf32>) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
