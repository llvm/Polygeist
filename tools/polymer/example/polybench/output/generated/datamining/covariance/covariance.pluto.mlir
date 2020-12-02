#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map2 = affine_map<(d0) -> (d0 * 32)>
#map3 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map4 = affine_map<(d0, d1) -> (d0 * 32, d1)>
module  {
  func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg6 = 0 to %0 {
      call @S0(%arg5, %arg6) : (memref<1200xf64>, index) -> ()
      %3 = alloca() : memref<1xf64>
      call @S1(%3, %arg5, %arg6) : (memref<1xf64>, memref<1200xf64>, index) -> ()
      affine.for %arg7 = 0 to %1 {
        call @S2(%arg5, %arg6, %arg3, %arg7, %3) : (memref<1200xf64>, index, memref<1400x1200xf64>, index, memref<1xf64>) -> ()
      }
      call @S3(%arg5, %arg6, %arg2) : (memref<1200xf64>, index, f64) -> ()
    }
    affine.for %arg6 = 0 to %1 {
      affine.for %arg7 = 0 to %0 {
        call @S4(%arg3, %arg6, %arg7, %arg5) : (memref<1400x1200xf64>, index, index, memref<1200xf64>) -> ()
      }
    }
    %2 = alloca() : memref<1xf64>
    call @S5(%2, %arg2) : (memref<1xf64>, f64) -> ()
    affine.for %arg6 = 0 to %0 {
      affine.for %arg7 = #map0(%arg6) to %0 {
        call @S6(%arg4, %arg6, %arg7) : (memref<1200x1200xf64>, index, index) -> ()
        %3 = alloca() : memref<1xf64>
        call @S7(%3, %arg4, %arg6, %arg7) : (memref<1xf64>, memref<1200x1200xf64>, index, index) -> ()
        affine.for %arg8 = 0 to %1 {
          call @S8(%arg4, %arg6, %arg7, %arg3, %arg8, %3) : (memref<1200x1200xf64>, index, index, memref<1400x1200xf64>, index, memref<1xf64>) -> ()
        }
        call @S9(%arg4, %arg6, %arg7, %2) : (memref<1200x1200xf64>, index, index, memref<1xf64>) -> ()
        call @S10(%arg4, %arg7, %arg6) : (memref<1200x1200xf64>, index, index) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<1200xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1)] : memref<1200xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: memref<1200xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<1200xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1200xf64>, %arg1: index, %arg2: memref<1400x1200xf64>, %arg3: index, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[0] : memref<1xf64>
    %1 = affine.load %arg2[symbol(%arg3), symbol(%arg1)] : memref<1400x1200xf64>
    %2 = addf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1)] : memref<1200xf64>
    return
  }
  func @S3(%arg0: memref<1200xf64>, %arg1: index, %arg2: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<1200xf64>
    %1 = divf %0, %arg2 : f64
    affine.store %1, %arg0[symbol(%arg1)] : memref<1200xf64>
    return
  }
  func @S4(%arg0: memref<1400x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1400x1200xf64>
    %1 = affine.load %arg3[symbol(%arg2)] : memref<1200xf64>
    %2 = subf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1400x1200xf64>
    return
  }
  func @S5(%arg0: memref<1xf64>, %arg1: f64) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    %0 = subf %arg1, %cst : f64
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S6(%arg0: memref<1200x1200xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1200x1200xf64>
    return
  }
  func @S7(%arg0: memref<1xf64>, %arg1: memref<1200x1200xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2), symbol(%arg3)] : memref<1200x1200xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S8(%arg0: memref<1200x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1400x1200xf64>, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg3[symbol(%arg4), symbol(%arg1)] : memref<1400x1200xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<1400x1200xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1200x1200xf64>
    return
  }
  func @S9(%arg0: memref<1200x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1200x1200xf64>
    %1 = affine.load %arg3[0] : memref<1xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1200x1200xf64>
    return
  }
  func @S10(%arg0: memref<1200x1200xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg2), symbol(%arg1)] : memref<1200x1200xf64>
    affine.store %0, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1200x1200xf64>
    return
  }
  func @kernel_covariance_new(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>) {
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = index_cast %arg1 : i32 to index
    %4 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to #map1()[%4] {
      affine.for %arg7 = #map0(%arg6) to #map1()[%4] {
        affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%4] {
          affine.for %arg9 = max #map4(%arg7, %arg8) to min #map3(%arg7)[%4] {
            call @S10(%arg4, %arg8, %arg9) : (memref<1200x1200xf64>, index, index) -> ()
            call @S6(%arg4, %arg8, %arg9) : (memref<1200x1200xf64>, index, index) -> ()
          }
        }
      }
    }
    affine.for %arg6 = 0 to %4 {
      affine.for %arg7 = #map0(%arg6) to %4 {
        call @S7(%0, %arg4, %arg6, %arg7) : (memref<1xf64>, memref<1200x1200xf64>, index, index) -> ()
        affine.for %arg8 = 0 to %3 {
          call @S8(%arg4, %arg6, %arg7, %arg3, %arg8, %0) : (memref<1200x1200xf64>, index, index, memref<1400x1200xf64>, index, memref<1xf64>) -> ()
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%4] {
      affine.for %arg7 = #map0(%arg6) to #map1()[%4] {
        affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%4] {
          affine.for %arg9 = max #map4(%arg7, %arg8) to min #map3(%arg7)[%4] {
            call @S9(%arg4, %arg8, %arg9, %1) : (memref<1200x1200xf64>, index, index, memref<1xf64>) -> ()
          }
        }
      }
    }
    call @S5(%1, %arg2) : (memref<1xf64>, f64) -> ()
    affine.for %arg6 = 0 to #map1()[%4] {
      affine.for %arg7 = #map2(%arg6) to min #map3(%arg6)[%4] {
        call @S0(%arg5, %arg7) : (memref<1200xf64>, index) -> ()
      }
    }
    affine.for %arg6 = 0 to %4 {
      call @S1(%2, %arg5, %arg6) : (memref<1xf64>, memref<1200xf64>, index) -> ()
      affine.for %arg7 = 0 to %3 {
        call @S2(%arg5, %arg6, %arg3, %arg7, %2) : (memref<1200xf64>, index, memref<1400x1200xf64>, index, memref<1xf64>) -> ()
      }
    }
    affine.for %arg6 = 0 to #map1()[%4] {
      affine.for %arg7 = #map2(%arg6) to min #map3(%arg6)[%4] {
        call @S3(%arg5, %arg7, %arg2) : (memref<1200xf64>, index, f64) -> ()
      }
    }
    affine.for %arg6 = 0 to #map1()[%4] {
      affine.for %arg7 = 0 to #map1()[%3] {
        affine.for %arg8 = #map2(%arg7) to min #map3(%arg7)[%3] {
          affine.for %arg9 = #map2(%arg6) to min #map3(%arg6)[%4] {
            call @S4(%arg3, %arg9, %arg8, %arg5) : (memref<1400x1200xf64>, index, index, memref<1200xf64>) -> ()
          }
        }
      }
    }
    return
  }
}

