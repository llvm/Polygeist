#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<()[s0, s1] -> (s0, s1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>


module {
  func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        call @S0(%arg5, %arg8, %arg9, %arg4) : (memref<1000x1100xf64>, index, index, f64) -> ()
      }
      affine.for %arg9 = 0 to %2 {
        %3 = alloca() : memref<1xf64>
        call @S1(%3, %arg3, %arg6, %arg8, %arg9) : (memref<1xf64>, f64, memref<1000x1200xf64>, index, index) -> ()
        affine.for %arg10 = 0 to %1 {
          call @S2(%arg5, %arg8, %arg10, %arg7, %arg9, %3) : (memref<1000x1100xf64>, index, index, memref<1200x1100xf64>, index, memref<1xf64>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<1000x1100xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1100xf64>
    %1 = mulf %0, %arg3 : f64
    affine.store %1, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1100xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: f64, %arg2: memref<1000x1200xf64>, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg2[symbol(%arg3), symbol(%arg4)] : memref<1000x1200xf64>
    %1 = mulf %arg1, %0 : f64
    affine.store %1, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1000x1100xf64>, %arg1: index, %arg2: index, %arg3: memref<1200x1100xf64>, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1100xf64>
    %1 = affine.load %arg5[0] : memref<1xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<1200x1100xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1100xf64>
    return
  }
  func @kernel_gemm_new(%arg0: f64, %arg1: memref<1000x1100xf64>, %arg2: memref<1000x1200xf64>, %arg3: memref<1200x1100xf64>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: f64) {
    %0 = index_cast %arg6 : i32 to index
    %1 = index_cast %arg5 : i32 to index
    %2 = index_cast %arg4 : i32 to index
    affine.for %arg8 = 0 to %2 {
      affine.for %arg9 = 0 to #map5()[%0] {
        affine.for %arg10 = #map3(%arg9) to min #map4(%arg9)[%0] {
          call @S0(%arg1, %arg8, %arg10, %arg7) : (memref<1000x1100xf64>, index, index, f64) -> ()
        }
      }
      affine.for %arg9 = 0 to %0 {
        %3 = alloca() : memref<1xf64>
        call @S1(%3, %arg0, %arg2, %arg8, %arg9) : (memref<1xf64>, f64, memref<1000x1200xf64>, index, index) -> ()
        affine.for %arg10 = 0 to #map5()[%1] {
          affine.for %arg11 = #map3(%arg10) to min #map4(%arg10)[%1] {
            call @S2(%arg1, %arg8, %arg9, %arg3, %arg11, %3) : (memref<1000x1100xf64>, index, index, memref<1200x1100xf64>, index, memref<1xf64>) -> ()
          }
        }
      }
    }
    return
  }
}
