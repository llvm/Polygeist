module  {
  func @kernel_mvt(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000x2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %0 {
      %1 = alloca() : memref<1xf64>
      call @S0(%1, %arg1, %arg6) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg7 = 0 to %0 {
        call @S1(%arg1, %arg6, %arg3, %arg7, %arg5, %1) : (memref<2000xf64>, index, memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      }
    }
    affine.for %arg6 = 0 to %0 {
      %1 = alloca() : memref<1xf64>
      call @S2(%1, %arg2, %arg6) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg7 = 0 to %0 {
        call @S3(%arg2, %arg6, %arg4, %arg7, %arg5, %1) : (memref<2000xf64>, index, memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>, %arg3: index, %arg4: memref<2000x2000xf64>, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg3)] : memref<2000x2000xf64>
    %2 = affine.load %arg2[symbol(%arg3)] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>, %arg3: index, %arg4: memref<2000x2000xf64>, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg4[symbol(%arg3), symbol(%arg1)] : memref<2000x2000xf64>
    %2 = affine.load %arg2[symbol(%arg3)] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @kernel_mvt_new(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000x2000xf64>) {
    %c0 = constant 0 : index
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %0 {
      %1 = alloca() : memref<1xf64>
      %2 = alloca() : memref<1xf64>
      call @S2(%2, %arg2, %arg6) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S3(%arg2, %arg6, %arg4, %c0, %arg5, %2) : (memref<2000xf64>, index, memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      call @S0(%1, %arg1, %arg6) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S1(%arg1, %arg6, %arg3, %c0, %arg5, %1) : (memref<2000xf64>, index, memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      affine.for %arg7 = 1 to %0 {
        call @S3(%arg2, %arg6, %arg4, %arg7, %arg5, %2) : (memref<2000xf64>, index, memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
        call @S1(%arg1, %arg6, %arg3, %arg7, %arg5, %1) : (memref<2000xf64>, index, memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      }
    }
    return
  }
}

