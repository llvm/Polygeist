module  {
  func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<1300x1300xf64>, %arg4: memref<1300x1300xf64>, %arg5: memref<1300xf64>, %arg6: memref<1300xf64>, %arg7: memref<1300xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg8 = 0 to %0 {
      call @S0(%arg5, %arg8) : (memref<1300xf64>, index) -> ()
      call @S1(%arg7, %arg8) : (memref<1300xf64>, index) -> ()
      %1 = alloca() : memref<1xf64>
      call @S2(%1, %arg5, %arg8) : (memref<1xf64>, memref<1300xf64>, index) -> ()
      %2 = alloca() : memref<1xf64>
      call @S3(%2, %arg7, %arg8) : (memref<1xf64>, memref<1300xf64>, index) -> ()
      affine.for %arg9 = 0 to %0 {
        call @S4(%arg5, %arg8, %1, %arg6, %arg9, %arg3) : (memref<1300xf64>, index, memref<1xf64>, memref<1300xf64>, index, memref<1300x1300xf64>) -> ()
        call @S5(%arg7, %arg8, %2, %arg6, %arg9, %arg4) : (memref<1300xf64>, index, memref<1xf64>, memref<1300xf64>, index, memref<1300x1300xf64>) -> ()
      }
      call @S6(%arg7, %arg8, %arg2, %arg1, %arg5) : (memref<1300xf64>, index, f64, f64, memref<1300xf64>) -> ()
    }
    return
  }
  func @S0(%arg0: memref<1300xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1] : memref<1300xf64>
    return
  }
  func @S1(%arg0: memref<1300xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1] : memref<1300xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: memref<1300xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2] : memref<1300xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<1300xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2] : memref<1300xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1300xf64>, %arg1: index, %arg2: memref<1xf64>, %arg3: memref<1300xf64>, %arg4: index, %arg5: memref<1300x1300xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[%arg1, %arg4] : memref<1300x1300xf64>
    %1 = affine.load %arg3[%arg4] : memref<1300xf64>
    %2 = mulf %0, %1 : f64
    %3 = affine.load %arg2[0] : memref<1xf64>
    %4 = addf %2, %3 : f64
    affine.store %4, %arg0[%arg1] : memref<1300xf64>
    return
  }
  func @S5(%arg0: memref<1300xf64>, %arg1: index, %arg2: memref<1xf64>, %arg3: memref<1300xf64>, %arg4: index, %arg5: memref<1300x1300xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[%arg1, %arg4] : memref<1300x1300xf64>
    %1 = affine.load %arg3[%arg4] : memref<1300xf64>
    %2 = mulf %0, %1 : f64
    %3 = affine.load %arg2[0] : memref<1xf64>
    %4 = addf %2, %3 : f64
    affine.store %4, %arg0[%arg1] : memref<1300xf64>
    return
  }
  func @S6(%arg0: memref<1300xf64>, %arg1: index, %arg2: f64, %arg3: f64, %arg4: memref<1300xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[%arg1] : memref<1300xf64>
    %1 = mulf %arg3, %0 : f64
    %2 = affine.load %arg0[%arg1] : memref<1300xf64>
    %3 = mulf %arg2, %2 : f64
    %4 = addf %1, %3 : f64
    affine.store %4, %arg0[%arg1] : memref<1300xf64>
    return
  }
}

