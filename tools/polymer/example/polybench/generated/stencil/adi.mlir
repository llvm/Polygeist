#map0 = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<(d0) -> (d0 - 1)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0) -> (d0)>
module  {
  func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
    %c0 = constant 0 : index
    %cst = constant 1.000000e+00 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %cst_1 = constant 0.000000e+00 : f64
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %0 = sitofp %arg1 : i32 to f64
    %1 = divf %cst, %0 : f64
    %2 = sitofp %arg0 : i32 to f64
    %3 = divf %cst, %2 : f64
    %4 = mulf %cst_0, %3 : f64
    %5 = mulf %1, %1 : f64
    %6 = divf %4, %5 : f64
    %7 = mulf %cst, %3 : f64
    %8 = divf %7, %5 : f64
    %9 = negf %6 : f64
    %10 = divf %9, %cst_0 : f64
    %11 = addf %cst, %6 : f64
    %12 = negf %8 : f64
    %13 = divf %12, %cst_0 : f64
    %14 = addf %cst, %8 : f64
    %15 = index_cast %arg0 : i32 to index
    %16 = index_cast %arg1 : i32 to index
    %17 = subi %16, %c1 : index
    %18 = negf %10 : f64
    %19 = negf %13 : f64
    %20 = mulf %cst_0, %13 : f64
    %21 = addf %cst, %20 : f64
    %22 = subi %16, %c2 : index
    %23 = addi %22, %c1 : index
    %24 = subi %23, %c1 : index
    %25 = mulf %cst_0, %10 : f64
    %26 = addf %cst, %25 : f64
    affine.for %arg6 = 1 to #map0()[%15] {
      affine.for %arg7 = 1 to #map1()[%16] {
        affine.store %cst, %arg3[%c0, %arg7] : memref<1000x1000xf64>
        affine.store %cst_1, %arg4[%arg7, %c0] : memref<1000x1000xf64>
        %27 = affine.load %arg3[%c0, %arg7] : memref<1000x1000xf64>
        affine.store %27, %arg5[%arg7, %c0] : memref<1000x1000xf64>
        affine.for %arg8 = 1 to #map1()[%16] {
          %28 = affine.apply #map2(%arg8)
          %29 = affine.load %arg4[%arg7, %28] : memref<1000x1000xf64>
          %30 = mulf %10, %29 : f64
          %31 = addf %30, %11 : f64
          %32 = divf %18, %31 : f64
          affine.store %32, %arg4[%arg7, %arg8] : memref<1000x1000xf64>
          %33 = affine.apply #map2(%arg7)
          %34 = affine.load %arg2[%arg8, %33] : memref<1000x1000xf64>
          %35 = mulf %19, %34 : f64
          %36 = affine.load %arg2[%arg8, %arg7] : memref<1000x1000xf64>
          %37 = mulf %21, %36 : f64
          %38 = addf %35, %37 : f64
          %39 = affine.apply #map3(%arg7)
          %40 = affine.load %arg2[%arg8, %39] : memref<1000x1000xf64>
          %41 = mulf %13, %40 : f64
          %42 = subf %38, %41 : f64
          %43 = affine.load %arg5[%arg7, %28] : memref<1000x1000xf64>
          %44 = mulf %10, %43 : f64
          %45 = subf %42, %44 : f64
          %46 = affine.load %arg4[%arg7, %28] : memref<1000x1000xf64>
          %47 = mulf %10, %46 : f64
          %48 = addf %47, %11 : f64
          %49 = divf %45, %48 : f64
          affine.store %49, %arg5[%arg7, %arg8] : memref<1000x1000xf64>
        }
        store %cst, %arg3[%17, %arg7] : memref<1000x1000xf64>
        affine.for %arg8 = 1 to #map1()[%16] {
          %28 = subi %arg8, %c1 : index
          %29 = subi %24, %28 : index
          %30 = affine.apply #map2(%arg8)
          %31 = affine.load %arg4[%arg7, %30] : memref<1000x1000xf64>
          %32 = affine.apply #map4(%arg8)
          %33 = affine.load %arg3[%32, %arg7] : memref<1000x1000xf64>
          %34 = mulf %31, %33 : f64
          %35 = affine.load %arg5[%arg7, %30] : memref<1000x1000xf64>
          %36 = addf %34, %35 : f64
          affine.store %36, %arg3[%30, %arg7] : memref<1000x1000xf64>
        }
      }
      affine.for %arg7 = 1 to #map1()[%16] {
        affine.store %cst, %arg2[%arg7, %c0] : memref<1000x1000xf64>
        affine.store %cst_1, %arg4[%arg7, %c0] : memref<1000x1000xf64>
        %27 = affine.load %arg2[%arg7, %c0] : memref<1000x1000xf64>
        affine.store %27, %arg5[%arg7, %c0] : memref<1000x1000xf64>
        affine.for %arg8 = 1 to #map1()[%16] {
          %28 = affine.apply #map2(%arg8)
          %29 = affine.load %arg4[%arg7, %28] : memref<1000x1000xf64>
          %30 = mulf %13, %29 : f64
          %31 = addf %30, %14 : f64
          %32 = divf %19, %31 : f64
          affine.store %32, %arg4[%arg7, %arg8] : memref<1000x1000xf64>
          %33 = affine.apply #map2(%arg7)
          %34 = affine.load %arg3[%33, %arg8] : memref<1000x1000xf64>
          %35 = mulf %18, %34 : f64
          %36 = affine.load %arg3[%arg7, %arg8] : memref<1000x1000xf64>
          %37 = mulf %26, %36 : f64
          %38 = addf %35, %37 : f64
          %39 = affine.apply #map3(%arg7)
          %40 = affine.load %arg3[%39, %arg8] : memref<1000x1000xf64>
          %41 = mulf %10, %40 : f64
          %42 = subf %38, %41 : f64
          %43 = affine.load %arg5[%arg7, %28] : memref<1000x1000xf64>
          %44 = mulf %13, %43 : f64
          %45 = subf %42, %44 : f64
          %46 = affine.load %arg4[%arg7, %28] : memref<1000x1000xf64>
          %47 = mulf %13, %46 : f64
          %48 = addf %47, %14 : f64
          %49 = divf %45, %48 : f64
          affine.store %49, %arg5[%arg7, %arg8] : memref<1000x1000xf64>
        }
        store %cst, %arg2[%arg7, %17] : memref<1000x1000xf64>
        affine.for %arg8 = 1 to #map1()[%16] {
          %28 = subi %arg8, %c1 : index
          %29 = subi %24, %28 : index
          %30 = affine.apply #map2(%arg8)
          %31 = affine.load %arg4[%arg7, %30] : memref<1000x1000xf64>
          %32 = affine.apply #map4(%arg8)
          %33 = affine.load %arg2[%arg7, %32] : memref<1000x1000xf64>
          %34 = mulf %31, %33 : f64
          %35 = affine.load %arg5[%arg7, %30] : memref<1000x1000xf64>
          %36 = addf %34, %35 : f64
          affine.store %36, %arg2[%arg7, %30] : memref<1000x1000xf64>
        }
      }
    }
    return
  }
}
