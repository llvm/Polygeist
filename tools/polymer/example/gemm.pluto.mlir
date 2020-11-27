#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>


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
    %0 = dim %arg2, %c1 : memref<?x?xf32>
    %1 = dim %arg0, %c1 : memref<?x?xf32>
    %c0 = constant 0 : index
    %2 = dim %arg2, %c0 : memref<?x?xf32>
    affine.for %arg5 = 0 to #map5()[%2] {
      affine.for %arg6 = 0 to #map5()[%1] {
        affine.for %arg7 = #map3(%arg5) to min #map4(%arg5)[%2] {
          affine.for %arg8 = #map3(%arg6) to min #map4(%arg6)[%1] {
            call @S0(%arg2, %arg7, %arg8, %arg4) : (memref<?x?xf32>, index, index, f32) -> ()
          }
        }
      }
    }
    affine.for %arg5 = 0 to #map5()[%2] {
      affine.for %arg6 = 0 to #map5()[%0] {
        affine.for %arg7 = 0 to #map5()[%1] {
          affine.for %arg8 = #map3(%arg5) to min #map4(%arg5)[%2] {
            affine.for %arg9 = #map3(%arg6) to min #map4(%arg6)[%0] {
              affine.for %arg10 = #map3(%arg7) to min #map4(%arg7)[%1] {
                call @S1(%arg2, %arg8, %arg9, %arg1, %arg10, %arg3, %arg0) : (memref<?x?xf32>, index, index, memref<?x?xf32>, index, f32, memref<?x?xf32>) -> ()
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
// [pluto] Diamond tiling not possible/useful
// [Pluto] After tiling:
// T(S1): (0, i0/32, i3/32, i0, i3, 0, 0)
// loop types (scalar, loop, loop, loop, loop, scalar, scalar)

// T(S2): (1, i0/32, i2/32, i1/32, i0, i2, i1)
// loop types (scalar, loop, loop, loop, loop, loop, loop)

// for (t2=0;t2<=floord(P0-1,32);t2++) {
//   for (t3=0;t3<=floord(P1-1,32);t3++) {
//     for (t4=32*t2;t4<=min(P0-1,32*t2+31);t4++) {
//       for (t5=32*t3;t5<=min(P1-1,32*t3+31);t5++) {
//         S0()
//       }
//     }
//   }
// }
// for (t2=0;t2<=floord(P0-1,32);t2++) {
//   for (t3=0;t3<=floord(P2-1,32);t3++) {
//     for (t4=0;t4<=floord(P1-1,32);t4++) {
//       for (t5=32*t2;t5<=min(P0-1,32*t2+31);t5++) {
//         for (t6=32*t3;t6<=min(P2-1,32*t3+31);t6++) {
//           for (t7=32*t4;t7<=min(P1-1,32*t4+31);t7++) {
//             S1()
//           }
//         }
//       }
//     }
//   }
// }
