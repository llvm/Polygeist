#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 - 1)>
module  {
  func @kernel_durbin(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
    %c0 = constant 0 : index
    %cst = constant 1.000000e+00 : f64
    %c1_i32 = constant 1 : i32
    %cst_0 = constant 0.000000e+00 : f64
    %c1 = constant 1 : index
    %0 = alloca() : memref<2000xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = affine.load %arg1[%c0] : memref<2000xf64>
    %5 = negf %4 : f64
    affine.store %5, %arg2[%c0] : memref<2000xf64>
    affine.store %cst, %2[%c0] : memref<1xf64>
    %6 = affine.load %arg1[%c0] : memref<2000xf64>
    %7 = negf %6 : f64
    affine.store %7, %1[%c0] : memref<1xf64>
    %8 = index_cast %arg0 : i32 to index
    %9 = sitofp %c1_i32 : i32 to f64
    %10 = affine.load %1[%c0] : memref<1xf64>
    %11 = affine.load %1[%c0] : memref<1xf64>
    %12 = mulf %10, %11 : f64
    %13 = subf %9, %12 : f64
    %14 = affine.load %2[%c0] : memref<1xf64>
    %15 = mulf %13, %14 : f64
    affine.store %15, %2[%c0] : memref<1xf64>
    affine.store %cst_0, %3[%c0] : memref<1xf64>
    %16 = affine.load %3[%c0] : memref<1xf64>
    %17 = affine.load %3[%c0] : memref<1xf64>
    affine.for %arg3 = 1 to %8 {
      affine.for %arg4 = 0 to #map0(%arg3) {
        %22 = subi %arg3, %arg4 : index
        %23 = affine.apply #map1(%arg4)
        %24 = affine.load %arg1[%23] : memref<2000xf64>
        %25 = affine.load %arg2[%arg4] : memref<2000xf64>
        %26 = mulf %24, %25 : f64
        %27 = addf %16, %26 : f64
        affine.store %27, %3[%c0] : memref<1xf64>
      }
      %18 = affine.load %arg1[%arg3] : memref<2000xf64>
      %19 = addf %18, %17 : f64
      %20 = negf %19 : f64
      %21 = divf %20, %15 : f64
      affine.store %21, %1[%c0] : memref<1xf64>
      affine.for %arg4 = 0 to #map0(%arg3) {
        %22 = affine.load %arg2[%arg4] : memref<2000xf64>
        %23 = subi %arg3, %arg4 : index
        %24 = affine.apply #map1(%arg4)
        %25 = affine.load %arg2[%24] : memref<2000xf64>
        %26 = mulf %21, %25 : f64
        %27 = addf %22, %26 : f64
        affine.store %27, %0[%arg4] : memref<2000xf64>
      }
      affine.for %arg4 = 0 to #map0(%arg3) {
        %22 = affine.load %0[%arg4] : memref<2000xf64>
        affine.store %22, %arg2[%arg4] : memref<2000xf64>
      }
      affine.store %21, %arg2[%arg3] : memref<2000xf64>
    }
    return
  }
}
