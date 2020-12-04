#map0 = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>
#map1 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map2 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32)>
#map6 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map7 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
module  {
  func @gemver(%arg0: f32, %arg1: f32, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xf32>, %arg10: memref<?xf32>) {
    %c0 = constant 0 : index
    %0 = dim %arg2, %c0 : memref<?x?xf32>
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %0 {
        call @S0(%arg2, %arg11, %arg12, %arg6, %arg5, %arg4, %arg3) : (memref<?x?xf32>, index, index, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
      }
    }
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %0 {
        call @S1(%arg8, %arg11, %arg9, %arg12, %arg1, %arg2) : (memref<?xf32>, index, memref<?xf32>, index, f32, memref<?x?xf32>) -> ()
      }
    }
    affine.for %arg11 = 0 to %0 {
      call @S2(%arg8, %arg11, %arg10) : (memref<?xf32>, index, memref<?xf32>) -> ()
    }
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %0 {
        call @S3(%arg7, %arg11, %arg8, %arg12, %arg0, %arg2) : (memref<?xf32>, index, memref<?xf32>, index, f32, memref<?x?xf32>) -> ()
      }
    }
    return
  }
  func private @S0(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<?x?xf32>
    %1 = affine.load %arg6[%arg1] : memref<?xf32>
    %2 = affine.load %arg5[%arg2] : memref<?xf32>
    %3 = mulf %1, %2 : f32
    %4 = addf %0, %3 : f32
    %5 = affine.load %arg4[%arg1] : memref<?xf32>
    %6 = affine.load %arg3[%arg2] : memref<?xf32>
    %7 = mulf %5, %6 : f32
    %8 = addf %4, %7 : f32
    affine.store %8, %arg0[%arg1, %arg2] : memref<?x?xf32>
    return
  }
  func private @S1(%arg0: memref<?xf32>, %arg1: index, %arg2: memref<?xf32>, %arg3: index, %arg4: f32, %arg5: memref<?x?xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg5[%arg3, %arg1] : memref<?x?xf32>
    %1 = mulf %arg4, %0 : f32
    %2 = affine.load %arg2[%arg3] : memref<?xf32>
    %3 = mulf %1, %2 : f32
    %4 = affine.load %arg0[%arg1] : memref<?xf32>
    %5 = addf %3, %4 : f32
    affine.store %5, %arg0[%arg1] : memref<?xf32>
    return
  }
  func private @S2(%arg0: memref<?xf32>, %arg1: index, %arg2: memref<?xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<?xf32>
    %1 = affine.load %arg2[%arg1] : memref<?xf32>
    %2 = addf %0, %1 : f32
    affine.store %2, %arg0[%arg1] : memref<?xf32>
    return
  }
  func private @S3(%arg0: memref<?xf32>, %arg1: index, %arg2: memref<?xf32>, %arg3: index, %arg4: f32, %arg5: memref<?x?xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<?xf32>
    %1 = affine.load %arg5[%arg1, %arg3] : memref<?x?xf32>
    %2 = mulf %arg4, %1 : f32
    %3 = affine.load %arg2[%arg3] : memref<?xf32>
    %4 = mulf %2, %3 : f32
    %5 = addf %0, %4 : f32
    affine.store %5, %arg0[%arg1] : memref<?xf32>
    return
  }
  func @gemver_new(%arg0: f32, %arg1: f32, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xf32>, %arg10: memref<?xf32>) {
    %c0 = constant 0 : index
    %0 = dim %arg2, %c0 : memref<?x?xf32>
    affine.for %arg11 = 0 to #map0()[%0] {
      affine.for %arg12 = max #map1(%arg11)[%0] to min #map2(%arg11)[%0] {
        affine.for %arg13 = #map3(%arg12) to min #map4(%arg12)[%0] {
          affine.for %arg14 = #map5(%arg11, %arg12) to min #map6(%arg11, %arg12)[%0] {
            call @S0(%arg2, %arg14, %arg13, %arg6, %arg5, %arg4, %arg3) : (memref<?x?xf32>, index, index, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
            call @S1(%arg8, %arg13, %arg9, %arg14, %arg1, %arg2) : (memref<?xf32>, index, memref<?xf32>, index, f32, memref<?x?xf32>) -> ()
          }
        }
      }
    }
    affine.for %arg11 = 0 to #map7()[%0] {
      affine.for %arg12 = #map3(%arg11) to min #map4(%arg11)[%0] {
        call @S2(%arg8, %arg12, %arg10) : (memref<?xf32>, index, memref<?xf32>) -> ()
      }
    }
    affine.for %arg11 = 0 to #map7()[%0] {
      affine.for %arg12 = 0 to #map7()[%0] {
        affine.for %arg13 = #map3(%arg11) to min #map4(%arg11)[%0] {
          affine.for %arg14 = #map3(%arg12) to min #map4(%arg12)[%0] {
            call @S3(%arg7, %arg13, %arg8, %arg14, %arg0, %arg2) : (memref<?xf32>, index, memref<?xf32>, index, f32, memref<?x?xf32>) -> ()
          }
        }
      }
    }
    return
  }
}

