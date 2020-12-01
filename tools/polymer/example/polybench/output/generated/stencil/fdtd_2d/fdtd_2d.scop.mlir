#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<() -> (1)>
#map3 = affine_map<()[s0] -> (s0 - 1)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0 - 1)>
#map7 = affine_map<(d0) -> (d0 + 1)>


module {
  func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %0 {
      %3 = alloca() : memref<1xf64>
      call @S0(%3, %arg6, %arg7) : (memref<1xf64>, memref<500xf64>, index) -> ()
      affine.for %arg8 = 0 to %1 {
        call @S1(%arg4, %arg8, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
      }
      affine.for %arg8 = 1 to %2 {
        affine.for %arg9 = 0 to %1 {
          call @S2(%arg4, %arg8, %arg9, %arg5) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.for %arg8 = 0 to %2 {
        affine.for %arg9 = 1 to %1 {
          call @S3(%arg3, %arg8, %arg9, %arg5) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.for %arg8 = 0 to #map3()[%2] {
        affine.for %arg9 = 0 to #map3()[%1] {
          call @S4(%arg5, %arg8, %arg9, %arg4, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf64>, %arg1: memref<500xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2] : memref<500xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0, %arg1] : memref<1000x1200xf64>
    return
  }
  func @S2(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %cst = constant 5.000000e-01 : f64
    %1 = affine.load %arg3[%arg1, %arg2] : memref<1000x1200xf64>
    %2 = affine.apply #map6(%arg1)
    %3 = affine.load %arg3[%2, %arg2] : memref<1000x1200xf64>
    %4 = subf %1, %3 : f64
    %5 = mulf %cst, %4 : f64
    %6 = subf %0, %5 : f64
    affine.store %6, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
  func @S3(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %cst = constant 5.000000e-01 : f64
    %1 = affine.load %arg3[%arg1, %arg2] : memref<1000x1200xf64>
    %2 = affine.apply #map6(%arg2)
    %3 = affine.load %arg3[%arg1, %2] : memref<1000x1200xf64>
    %4 = subf %1, %3 : f64
    %5 = mulf %cst, %4 : f64
    %6 = subf %0, %5 : f64
    affine.store %6, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
  func @S4(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %cst = constant 0.69999999999999996 : f64
    %1 = affine.apply #map7(%arg2)
    %2 = affine.load %arg4[%arg1, %1] : memref<1000x1200xf64>
    %3 = affine.load %arg4[%arg1, %arg2] : memref<1000x1200xf64>
    %4 = subf %2, %3 : f64
    %5 = affine.apply #map7(%arg1)
    %6 = affine.load %arg3[%5, %arg2] : memref<1000x1200xf64>
    %7 = addf %4, %6 : f64
    %8 = affine.load %arg3[%arg1, %arg2] : memref<1000x1200xf64>
    %9 = subf %7, %8 : f64
    %10 = mulf %cst, %9 : f64
    %11 = subf %0, %10 : f64
    affine.store %11, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
}
