#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> (32, s0)>
#map2 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<(d0) -> (32, d0)>
#map6 = affine_map<(d0) -> ((d0 - 1) floordiv 32 + 1)>
#map7 = affine_map<(d0, d1) -> (d0, d1 * 32 + 32)>
#set0 = affine_set<(d0) : (d0 - 2 >= 0)>
#set1 = affine_set<(d0) : (d0 - 1 == 0)>
module  {
  func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg0 : i32 to index
    %2 = index_cast %arg1 : i32 to index
    call @S0(%0) : (memref<1xf64>) -> ()
    %3 = alloca() : memref<1xf64>
    call @S1(%3, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    %4 = alloca() : memref<1xf64>
    call @S2(%4, %arg2, %0) : (memref<1xf64>, f64, memref<1xf64>) -> ()
    affine.for %arg7 = 0 to %1 {
      %5 = alloca() : memref<1xf64>
      call @S3(%5, %arg5, %arg7) : (memref<1xf64>, memref<1000x1000xf64>, index) -> ()
      affine.for %arg8 = 0 to %2 {
        %6 = alloca() : memref<1xf64>
        call @S4(%6, %arg2, %arg6, %arg7, %arg8) : (memref<1xf64>, f64, memref<1000x1200xf64>, index, index) -> ()
        affine.for %arg9 = 0 to #map0(%arg7) {
          call @S5(%arg4, %arg9, %arg8, %arg5, %arg7, %6) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, memref<1xf64>) -> ()
          call @S6(%0, %arg5, %arg7, %arg9, %arg6, %arg8, %3) : (memref<1xf64>, memref<1000x1000xf64>, index, index, memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        }
        call @S7(%arg4, %arg7, %arg8, %4, %5, %arg2, %arg6, %arg3) : (memref<1000x1200xf64>, index, index, memref<1xf64>, memref<1xf64>, f64, memref<1000x1200xf64>, f64) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %c0_i32 = constant 0 : i32
    %0 = sitofp %c0_i32 : i32 to f64
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: f64, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    %1 = mulf %arg1, %0 : f64
    affine.store %1, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<1000x1000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2), symbol(%arg2)] : memref<1000x1000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: f64, %arg2: memref<1000x1200xf64>, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg2[symbol(%arg3), symbol(%arg4)] : memref<1000x1200xf64>
    %1 = mulf %arg1, %0 : f64
    affine.store %1, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1000xf64>, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %1 = affine.load %arg5[0] : memref<1xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg1)] : memref<1000x1000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    return
  }
  func @S6(%arg0: memref<1xf64>, %arg1: memref<1000x1000xf64>, %arg2: index, %arg3: index, %arg4: memref<1000x1200xf64>, %arg5: index, %arg6: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg6[0] : memref<1xf64>
    %1 = affine.load %arg4[symbol(%arg3), symbol(%arg5)] : memref<1000x1200xf64>
    %2 = affine.load %arg1[symbol(%arg2), symbol(%arg3)] : memref<1000x1000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func @S7(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1xf64>, %arg5: f64, %arg6: memref<1000x1200xf64>, %arg7: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %1 = mulf %arg7, %0 : f64
    %2 = affine.load %arg6[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %3 = mulf %arg5, %2 : f64
    %4 = affine.load %arg4[0] : memref<1xf64>
    %5 = mulf %3, %4 : f64
    %6 = addf %1, %5 : f64
    %7 = affine.load %arg3[0] : memref<1xf64>
    %8 = addf %6, %7 : f64
    affine.store %8, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    return
  }
  func @kernel_symm_new(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = alloca() : memref<1xf64>
    %5 = index_cast %arg1 : i32 to index
    %6 = index_cast %arg0 : i32 to index
    call @S0(%4) : (memref<1xf64>) -> ()
    call @S2(%2, %arg2, %4) : (memref<1xf64>, f64, memref<1xf64>) -> ()
    affine.for %arg7 = 0 to %6 {
      call @S7(%arg4, %arg7, %c0, %2, %1, %arg2, %arg6, %arg3) : (memref<1000x1200xf64>, index, index, memref<1xf64>, memref<1xf64>, f64, memref<1000x1200xf64>, f64) -> ()
      call @S3(%1, %arg5, %arg7) : (memref<1xf64>, memref<1000x1000xf64>, index) -> ()
      affine.for %arg8 = 1 to min #map1()[%5] {
        call @S7(%arg4, %arg7, %arg8, %2, %1, %arg2, %arg6, %arg3) : (memref<1000x1200xf64>, index, index, memref<1xf64>, memref<1xf64>, f64, memref<1000x1200xf64>, f64) -> ()
      }
      affine.for %arg8 = 1 to #map2()[%5] {
        affine.for %arg9 = #map3(%arg8) to min #map4(%arg8)[%5] {
          call @S7(%arg4, %arg7, %arg9, %2, %1, %arg2, %arg6, %arg3) : (memref<1000x1200xf64>, index, index, memref<1xf64>, memref<1xf64>, f64, memref<1000x1200xf64>, f64) -> ()
        }
      }
    }
    affine.for %arg7 = 0 to %5 {
      call @S4(%0, %arg2, %arg6, %c0, %arg7) : (memref<1xf64>, f64, memref<1000x1200xf64>, index, index) -> ()
    }
    affine.for %arg7 = 1 to %6 {
      affine.for %arg8 = 0 to %5 {
        call @S5(%arg4, %arg7, %arg8, %arg5, %c0, %0) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, memref<1xf64>) -> ()
        affine.if #set0(%arg7) {
          call @S4(%0, %arg2, %arg6, %arg7, %arg8) : (memref<1xf64>, f64, memref<1000x1200xf64>, index, index) -> ()
          call @S5(%arg4, %arg7, %arg8, %arg5, %c1, %0) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, memref<1xf64>) -> ()
        }
        affine.if #set1(%arg7) {
          call @S4(%0, %arg2, %arg6, %c1, %arg8) : (memref<1xf64>, f64, memref<1000x1200xf64>, index, index) -> ()
        }
        affine.for %arg9 = 2 to min #map5(%arg7) {
          call @S5(%arg4, %arg7, %arg8, %arg5, %arg9, %0) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, memref<1xf64>) -> ()
        }
        affine.for %arg9 = 1 to #map6(%arg7) {
          affine.for %arg10 = #map3(%arg9) to min #map7(%arg7, %arg9) {
            call @S5(%arg4, %arg7, %arg8, %arg5, %arg10, %0) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, memref<1xf64>) -> ()
          }
        }
      }
    }
    call @S1(%3, %4) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg7 = 1 to %6 {
      affine.for %arg8 = 0 to %5 {
        affine.for %arg9 = 0 to #map0(%arg7) {
          call @S6(%4, %arg5, %arg7, %arg8, %arg6, %arg9, %3) : (memref<1xf64>, memref<1000x1000xf64>, index, index, memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        }
      }
    }
    return
  }
}

