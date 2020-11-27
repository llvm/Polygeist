#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 31)>
#map5 = affine_map<()[s0] -> (31, s0 - 1)>
#map6 = affine_map<() -> (1)>
#map7 = affine_map<()[s0] -> ((s0 - 1) floordiv 32)>


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
  func @gemm_new(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: f32, %arg4: f32) {
    %c1 = constant 1 : index
    %0 = dim %arg0, %c1 : memref<?x?xf32>
    %1 = dim %arg1, %c1 : memref<?x?xf32>
    %c0 = constant 0 : index
    %2 = dim %arg0, %c0 : memref<?x?xf32>
    affine.for %arg5 = 0 to #map7()[%2] {
      affine.for %arg6 = 0 to #map7()[%1] {
        affine.for %arg7 = #map3(%arg5) to min #map4(%arg5)[%2] {
          affine.for %arg8 = #map3(%arg6) to min #map4(%arg6)[%1] {
            call @S0(%arg0, %arg7, %arg8, %arg4) : (memref<?x?xf32>, index, index, f32) -> ()
          }
          affine.for %arg8 = 0 to min #map5()[%0] {
            affine.for %arg9 = #map3(%arg6) to min #map4(%arg6)[%1] {
              call @S1(%arg0, %arg7, %arg8, %arg2, %arg9, %arg3, %arg1) : (memref<?x?xf32>, index, index, memref<?x?xf32>, index, f32, memref<?x?xf32>) -> ()
            }
          }
        }
        affine.for %arg7 = 1 to #map7()[%0] {
          affine.for %arg8 = #map3(%arg5) to min #map4(%arg5)[%2] {
            affine.for %arg9 = #map3(%arg7) to min #map4(%arg7)[%0] {
              affine.for %arg10 = #map3(%arg6) to min #map4(%arg6)[%1] {
                call @S1(%arg0, %arg8, %arg9, %arg2, %arg10, %arg3, %arg1) : (memref<?x?xf32>, index, index, memref<?x?xf32>, index, f32, memref<?x?xf32>) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
// [pluto] compute_deps (isl)
// [Pluto] After tiling:
// T(S1): (0, i0/32, 0/32, i3/32, 0/32, 0, i0, 0, i3, 0, 0, 0)
// loop types (scalar, loop, loop, loop, loop, scalar, loop, scalar, loop, scalar, scalar, scalar)

// T(S2): (0, i0/32, zT4, i1/32, 0/32, i2/32, i0, 1, i1, 0, i2, 0)
// loop types (scalar, loop, loop, loop, loop, loop, loop, scalar, loop, scalar, loop, scalar)

// [Pluto] After intra-tile optimize
// T(S1): (0, i0/32, 0/32, i3/32, 0/32, 0, i0, 0, i3, 0, 0, 0)
// loop types (scalar, loop, loop, loop, loop, scalar, loop, scalar, loop, scalar, scalar, scalar)

// T(S2): (0, i0/32, zT4, i1/32, 0/32, i2/32, i0, 1, 0, i2, i1, 0)
// loop types (scalar, loop, loop, loop, loop, loop, loop, scalar, loop, scalar, loop, scalar)

