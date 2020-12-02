#map = affine_map<(d0) -> (d0)>
module  {
  func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg0 : i32 to index
    %2 = affine.load %0[%c0] : memref<1xf64>
    %3 = affine.load %0[%c0] : memref<1xf64>
    %4 = affine.load %0[%c0] : memref<1xf64>
    %5 = affine.load %0[%c0] : memref<1xf64>
    affine.for %arg5 = 0 to %1 {
      affine.for %arg6 = 0 to #map(%arg5) {
        %13 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
        affine.store %13, %0[%c0] : memref<1xf64>
        affine.for %arg7 = 0 to #map(%arg6) {
          %16 = affine.load %arg1[%arg5, %arg7] : memref<2000x2000xf64>
          %17 = affine.load %arg1[%arg7, %arg6] : memref<2000x2000xf64>
          %18 = mulf %16, %17 : f64
          %19 = subf %2, %18 : f64
          affine.store %19, %0[%c0] : memref<1xf64>
        }
        %14 = affine.load %arg1[%arg6, %arg6] : memref<2000x2000xf64>
        %15 = divf %3, %14 : f64
        affine.store %15, %arg1[%arg5, %arg6] : memref<2000x2000xf64>
      }
      affine.for %arg6 = #map(%arg5) to %1 {
        %13 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
        affine.store %13, %0[%c0] : memref<1xf64>
        affine.for %arg7 = 0 to #map(%arg5) {
          %14 = affine.load %arg1[%arg5, %arg7] : memref<2000x2000xf64>
          %15 = affine.load %arg1[%arg7, %arg6] : memref<2000x2000xf64>
          %16 = mulf %14, %15 : f64
          %17 = subf %4, %16 : f64
          affine.store %17, %0[%c0] : memref<1xf64>
        }
        affine.store %5, %arg1[%arg5, %arg6] : memref<2000x2000xf64>
      }
    }
    %6 = affine.load %0[%c0] : memref<1xf64>
    %7 = affine.load %0[%c0] : memref<1xf64>
    affine.for %arg5 = 0 to %1 {
      %13 = affine.load %arg2[%arg5] : memref<2000xf64>
      affine.store %13, %0[%c0] : memref<1xf64>
      affine.for %arg6 = 0 to #map(%arg5) {
        %14 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
        %15 = affine.load %arg4[%arg6] : memref<2000xf64>
        %16 = mulf %14, %15 : f64
        %17 = subf %6, %16 : f64
        affine.store %17, %0[%c0] : memref<1xf64>
      }
      affine.store %7, %arg4[%arg5] : memref<2000xf64>
    }
    %8 = subi %1, %c1 : index
    %9 = addi %8, %c1 : index
    %10 = subi %9, %c1 : index
    %11 = affine.load %0[%c0] : memref<1xf64>
    %12 = affine.load %0[%c0] : memref<1xf64>
    affine.for %arg5 = 0 to %1 {
      %13 = affine.apply #map(%arg5)
      %14 = affine.load %arg4[%13] : memref<2000xf64>
      affine.store %14, %0[%c0] : memref<1xf64>
      affine.for %arg6 = 1 to %1 {
        %17 = affine.load %arg1[%13, %arg6] : memref<2000x2000xf64>
        %18 = affine.load %arg3[%arg6] : memref<2000xf64>
        %19 = mulf %17, %18 : f64
        %20 = subf %11, %19 : f64
        affine.store %20, %0[%c0] : memref<1xf64>
      }
      %15 = affine.load %arg1[%13, %13] : memref<2000x2000xf64>
      %16 = divf %12, %15 : f64
      affine.store %16, %arg3[%13] : memref<2000xf64>
    }
    return
  }
}

