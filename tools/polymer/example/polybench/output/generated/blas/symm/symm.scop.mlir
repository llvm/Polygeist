#map0 = affine_map<() -> (0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<()[s0] -> (s0)>
#map3 = affine_map<(d0) -> (d0, d0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>


module {
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
        affine.for %arg9 = 0 to #map1(%arg7) {
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
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: f64, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg2[%c0] : memref<1xf64>
    %1 = mulf %arg1, %0 : f64
    affine.store %1, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<1000x1000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg2] : memref<1000x1000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: f64, %arg2: memref<1000x1200xf64>, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg2[%arg3, %arg4] : memref<1000x1200xf64>
    %1 = mulf %arg1, %0 : f64
    affine.store %1, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1000xf64>, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %1 = affine.load %arg5[0] : memref<1xf64>
    %2 = affine.load %arg3[%arg4, %arg1] : memref<1000x1000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
  func @S6(%arg0: memref<1xf64>, %arg1: memref<1000x1000xf64>, %arg2: index, %arg3: index, %arg4: memref<1000x1200xf64>, %arg5: index, %arg6: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg6[0] : memref<1xf64>
    %1 = affine.load %arg4[%arg3, %arg5] : memref<1000x1200xf64>
    %2 = affine.load %arg1[%arg2, %arg3] : memref<1000x1000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    %c0 = constant 0 : index
    affine.store %4, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S7(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1xf64>, %arg5: f64, %arg6: memref<1000x1200xf64>, %arg7: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %1 = mulf %arg7, %0 : f64
    %2 = affine.load %arg6[%arg1, %arg2] : memref<1000x1200xf64>
    %3 = mulf %arg5, %2 : f64
    %4 = affine.load %arg4[0] : memref<1xf64>
    %5 = mulf %3, %4 : f64
    %6 = addf %1, %5 : f64
    %7 = affine.load %arg3[0] : memref<1xf64>
    %8 = addf %6, %7 : f64
    affine.store %8, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
}
