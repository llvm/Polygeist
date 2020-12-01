#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> (d0 * 32)>
#map5 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map6 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>


module {
  func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<800x900xf64>, %arg7: memref<800x1100xf64>, %arg8: memref<1100x900xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<800x1200xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %1 {
        call @S0(%arg6, %arg11, %arg12) : (memref<800x900xf64>, index, index) -> ()
        %4 = alloca() : memref<1xf64>
        call @S1(%4, %arg6, %arg11, %arg12) : (memref<1xf64>, memref<800x900xf64>, index, index) -> ()
        affine.for %arg13 = 0 to %2 {
          call @S2(%arg6, %arg11, %arg12, %arg8, %arg13, %arg4, %arg7, %4) : (memref<800x900xf64>, index, index, memref<1100x900xf64>, index, f64, memref<800x1100xf64>, memref<1xf64>) -> ()
        }
      }
    }
    %3 = index_cast %arg3 : i32 to index
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %3 {
        call @S3(%arg10, %arg11, %arg12, %arg5) : (memref<800x1200xf64>, index, index, f64) -> ()
        %4 = alloca() : memref<1xf64>
        call @S4(%4, %arg10, %arg11, %arg12) : (memref<1xf64>, memref<800x1200xf64>, index, index) -> ()
        affine.for %arg13 = 0 to %1 {
          call @S5(%arg10, %arg11, %arg12, %arg9, %arg13, %arg6, %4) : (memref<800x1200xf64>, index, index, memref<900x1200xf64>, index, memref<800x900xf64>, memref<1xf64>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<800x900xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, %arg2] : memref<800x900xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: memref<800x900xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<800x900xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<800x900xf64>, %arg1: index, %arg2: index, %arg3: memref<1100x900xf64>, %arg4: index, %arg5: f64, %arg6: memref<800x1100xf64>, %arg7: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg7[0] : memref<1xf64>
    %1 = affine.load %arg6[%arg1, %arg4] : memref<800x1100xf64>
    %2 = mulf %arg5, %1 : f64
    %3 = affine.load %arg3[%arg4, %arg2] : memref<1100x900xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[%arg1, %arg2] : memref<800x900xf64>
    return
  }
  func @S3(%arg0: memref<800x1200xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<800x1200xf64>
    %1 = mulf %0, %arg3 : f64
    affine.store %1, %arg0[%arg1, %arg2] : memref<800x1200xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: memref<800x1200xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<800x1200xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<800x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<900x1200xf64>, %arg4: index, %arg5: memref<800x900xf64>, %arg6: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg6[0] : memref<1xf64>
    %1 = affine.load %arg5[%arg1, %arg4] : memref<800x900xf64>
    %2 = affine.load %arg3[%arg4, %arg2] : memref<900x1200xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<800x1200xf64>
    return
  }
  func @kernel_2mm_new(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1100xf64>, %arg7: memref<1100x900xf64>, %arg8: memref<800x1200xf64>, %arg9: memref<900x1200xf64>, %arg10: i32) {
    %0 = index_cast %arg4 : i32 to index
    %1 = index_cast %arg3 : i32 to index
    %2 = index_cast %arg0 : i32 to index
    affine.for %arg11 = 0 to #map6()[%1] {
      affine.for %arg12 = 0 to #map6()[%2] {
        %3 = alloca() : memref<1xf64>
        %4 = alloca() : memref<1xf64>
        affine.for %arg13 = #map4(%arg11) to min #map5(%arg11)[%1] {
          affine.for %arg14 = #map4(%arg12) to min #map5(%arg12)[%2] {
            %5 = affine.apply #map3(%arg13)
            %6 = affine.apply #map3(%arg14)
            call @S3(%arg8, %5, %6, %arg2) : (memref<800x1200xf64>, index, index, f64) -> ()
          }
        }
      }
      affine.for %arg12 = 0 to #map6()[%2] {
        affine.for %arg13 = #map4(%arg11) to min #map5(%arg11)[%1] {
          affine.for %arg14 = #map4(%arg12) to min #map5(%arg12)[%2] {
            %3 = affine.apply #map3(%arg13)
            %4 = affine.apply #map3(%arg14)
            call @S0(%arg5, %3, %4) : (memref<800x900xf64>, index, index) -> ()
          }
        }
      }
    }
    affine.for %arg11 = 0 to %1 {
      affine.for %arg12 = 0 to %2 {
        %3 = alloca() : memref<1xf64>
        %4 = affine.apply #map3(%arg11)
        %5 = affine.apply #map3(%arg12)
        call @S1(%3, %arg5, %4, %5) : (memref<1xf64>, memref<800x900xf64>, index, index) -> ()
        affine.for %arg13 = 0 to %0 {
          %6 = alloca() : memref<1xf64>
          %7 = affine.apply #map3(%arg13)
          call @S2(%arg5, %4, %5, %arg7, %7, %arg1, %arg6, %6) : (memref<800x900xf64>, index, index, memref<1100x900xf64>, index, f64, memref<800x1100xf64>, memref<1xf64>) -> ()
        }
      }
      affine.for %arg12 = 0 to %2 {
        %3 = alloca() : memref<1xf64>
        %4 = affine.apply #map3(%arg11)
        %5 = affine.apply #map3(%arg12)
        call @S4(%3, %arg8, %4, %5) : (memref<1xf64>, memref<800x1200xf64>, index, index) -> ()
        affine.for %arg13 = 0 to %0 {
          %6 = alloca() : memref<1xf64>
          %7 = affine.apply #map3(%arg13)
          call @S5(%arg8, %4, %5, %arg9, %7, %arg5, %6) : (memref<800x1200xf64>, index, index, memref<900x1200xf64>, index, memref<800x900xf64>, memref<1xf64>) -> ()
        }
      }
    }
    return
  }
}
