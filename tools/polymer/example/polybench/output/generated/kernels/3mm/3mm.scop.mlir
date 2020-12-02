module  {
  func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1000xf64>, %arg7: memref<1000x900xf64>, %arg8: memref<900x1100xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<1200x1100xf64>, %arg11: memref<800x1100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %1 {
        call @S0(%arg5, %arg12, %arg13) : (memref<800x900xf64>, index, index) -> ()
        %5 = alloca() : memref<1xf64>
        call @S1(%5, %arg5, %arg12, %arg13) : (memref<1xf64>, memref<800x900xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %2 {
          call @S2(%arg5, %arg12, %arg13, %arg7, %arg14, %arg6, %5) : (memref<800x900xf64>, index, index, memref<1000x900xf64>, index, memref<800x1000xf64>, memref<1xf64>) -> ()
        }
      }
    }
    %3 = index_cast %arg3 : i32 to index
    %4 = index_cast %arg4 : i32 to index
    affine.for %arg12 = 0 to %1 {
      affine.for %arg13 = 0 to %3 {
        call @S3(%arg8, %arg12, %arg13) : (memref<900x1100xf64>, index, index) -> ()
        %5 = alloca() : memref<1xf64>
        call @S4(%5, %arg8, %arg12, %arg13) : (memref<1xf64>, memref<900x1100xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %4 {
          call @S5(%arg8, %arg12, %arg13, %arg10, %arg14, %arg9, %5) : (memref<900x1100xf64>, index, index, memref<1200x1100xf64>, index, memref<900x1200xf64>, memref<1xf64>) -> ()
        }
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %3 {
        call @S6(%arg11, %arg12, %arg13) : (memref<800x1100xf64>, index, index) -> ()
        %5 = alloca() : memref<1xf64>
        call @S7(%5, %arg11, %arg12, %arg13) : (memref<1xf64>, memref<800x1100xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %1 {
          call @S8(%arg11, %arg12, %arg13, %arg8, %arg14, %arg5, %5) : (memref<800x1100xf64>, index, index, memref<900x1100xf64>, index, memref<800x900xf64>, memref<1xf64>) -> ()
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
  func @S2(%arg0: memref<800x900xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x900xf64>, %arg4: index, %arg5: memref<800x1000xf64>, %arg6: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg6[0] : memref<1xf64>
    %1 = affine.load %arg5[%arg1, %arg4] : memref<800x1000xf64>
    %2 = affine.load %arg3[%arg4, %arg2] : memref<1000x900xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<800x900xf64>
    return
  }
  func @S3(%arg0: memref<900x1100xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, %arg2] : memref<900x1100xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: memref<900x1100xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<900x1100xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<900x1100xf64>, %arg1: index, %arg2: index, %arg3: memref<1200x1100xf64>, %arg4: index, %arg5: memref<900x1200xf64>, %arg6: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg6[0] : memref<1xf64>
    %1 = affine.load %arg5[%arg1, %arg4] : memref<900x1200xf64>
    %2 = affine.load %arg3[%arg4, %arg2] : memref<1200x1100xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<900x1100xf64>
    return
  }
  func @S6(%arg0: memref<800x1100xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, %arg2] : memref<800x1100xf64>
    return
  }
  func @S7(%arg0: memref<1xf64>, %arg1: memref<800x1100xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<800x1100xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S8(%arg0: memref<800x1100xf64>, %arg1: index, %arg2: index, %arg3: memref<900x1100xf64>, %arg4: index, %arg5: memref<800x900xf64>, %arg6: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg6[0] : memref<1xf64>
    %1 = affine.load %arg5[%arg1, %arg4] : memref<800x900xf64>
    %2 = affine.load %arg3[%arg4, %arg2] : memref<900x1100xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<800x1100xf64>
    return
  }
}

