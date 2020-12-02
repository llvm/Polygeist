#map0 = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<(d0) -> (d0 - 1)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0) -> (d0)>
module  {
  func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
    %cst = constant 1.000000e+00 : f64
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %0 = alloca() : memref<1xf64>
    call @S0(%0, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %1 = alloca() : memref<1xf64>
    call @S1(%1, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %2 = alloca() : memref<1xf64>
    call @S2(%2, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %3 = alloca() : memref<1xf64>
    call @S3(%3, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %4 = index_cast %arg0 : i32 to index
    %5 = index_cast %arg1 : i32 to index
    %6 = subi %5, %c1 : index
    %7 = alloca() : memref<1xf64>
    call @S4(%7, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %8 = alloca() : memref<1xf64>
    call @S5(%8, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %9 = alloca() : memref<1xf64>
    call @S6(%9, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    %10 = subi %5, %c2 : index
    %11 = addi %10, %c1 : index
    %12 = subi %11, %c1 : index
    %13 = alloca() : memref<1xf64>
    call @S7(%13, %arg1, %arg0) : (memref<1xf64>, i32, i32) -> ()
    affine.for %arg6 = 1 to #map0()[%4] {
      affine.for %arg7 = 1 to #map1()[%5] {
        call @S8(%arg3, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S9(%arg4, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S10(%arg5, %arg7, %arg3) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        affine.for %arg8 = 1 to #map1()[%5] {
          call @S11(%arg4, %arg7, %arg8, %1, %0, %7) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>, memref<1xf64>) -> ()
          call @S12(%arg5, %arg7, %arg8, %1, %arg4, %0, %arg2, %2, %9, %8) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1xf64>, memref<1xf64>) -> ()
        }
        store %cst, %arg3[%6, %arg7] : memref<1000x1000xf64>
        affine.for %arg8 = 1 to #map1()[%5] {
          %14 = subi %arg8, %c1 : index
          %15 = subi %12, %14 : index
          call @S13(%arg3, %arg7, %arg8, %arg5, %arg4) : (memref<1000x1000xf64>, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
        }
      }
      affine.for %arg7 = 1 to #map1()[%5] {
        call @S14(%arg2, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S15(%arg4, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S16(%arg5, %arg7, %arg2) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        affine.for %arg8 = 1 to #map1()[%5] {
          call @S17(%arg4, %arg7, %arg8, %3, %2, %8) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>, memref<1xf64>) -> ()
          call @S18(%arg5, %arg7, %arg8, %3, %arg4, %2, %arg3, %0, %13, %7) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1xf64>, memref<1xf64>) -> ()
        }
        store %cst, %arg2[%arg7, %6] : memref<1000x1000xf64>
        affine.for %arg8 = 1 to #map1()[%5] {
          %14 = subi %arg8, %c1 : index
          %15 = subi %12, %14 : index
          call @S19(%arg2, %arg7, %arg8, %arg5, %arg4) : (memref<1000x1000xf64>, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = sitofp %arg1 : i32 to f64
    %3 = divf %cst, %2 : f64
    %4 = mulf %3, %3 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %5 = mulf %cst_0, %1 : f64
    %6 = divf %5, %4 : f64
    %7 = negf %6 : f64
    %8 = divf %7, %cst_0 : f64
    affine.store %8, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %0 = sitofp %arg2 : i32 to f64
    %cst_0 = constant 1.000000e+00 : f64
    %1 = divf %cst_0, %0 : f64
    %2 = mulf %cst, %1 : f64
    %3 = sitofp %arg1 : i32 to f64
    %4 = divf %cst_0, %3 : f64
    %5 = mulf %4, %4 : f64
    %6 = divf %2, %5 : f64
    %7 = addf %cst_0, %6 : f64
    affine.store %7, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = mulf %cst, %1 : f64
    %3 = sitofp %arg1 : i32 to f64
    %4 = divf %cst, %3 : f64
    %5 = mulf %4, %4 : f64
    %6 = divf %2, %5 : f64
    %7 = negf %6 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %8 = divf %7, %cst_0 : f64
    affine.store %8, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = mulf %cst, %1 : f64
    %3 = sitofp %arg1 : i32 to f64
    %4 = divf %cst, %3 : f64
    %5 = mulf %4, %4 : f64
    %6 = divf %2, %5 : f64
    %7 = addf %cst, %6 : f64
    affine.store %7, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = sitofp %arg1 : i32 to f64
    %3 = divf %cst, %2 : f64
    %4 = mulf %3, %3 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %5 = mulf %cst_0, %1 : f64
    %6 = divf %5, %4 : f64
    %7 = negf %6 : f64
    %8 = divf %7, %cst_0 : f64
    %9 = negf %8 : f64
    affine.store %9, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = mulf %cst, %1 : f64
    %3 = sitofp %arg1 : i32 to f64
    %4 = divf %cst, %3 : f64
    %5 = mulf %4, %4 : f64
    %6 = divf %2, %5 : f64
    %7 = negf %6 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %8 = divf %7, %cst_0 : f64
    %9 = negf %8 : f64
    affine.store %9, %arg0[0] : memref<1xf64>
    return
  }
  func @S6(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = mulf %cst, %1 : f64
    %3 = sitofp %arg1 : i32 to f64
    %4 = divf %cst, %3 : f64
    %5 = mulf %4, %4 : f64
    %6 = divf %2, %5 : f64
    %7 = negf %6 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %8 = divf %7, %cst_0 : f64
    %9 = mulf %cst_0, %8 : f64
    %10 = addf %cst, %9 : f64
    affine.store %10, %arg0[0] : memref<1xf64>
    return
  }
  func @S7(%arg0: memref<1xf64>, %arg1: i32, %arg2: i32) attributes {scop.stmt} {
    %0 = sitofp %arg2 : i32 to f64
    %cst = constant 1.000000e+00 : f64
    %1 = divf %cst, %0 : f64
    %2 = sitofp %arg1 : i32 to f64
    %3 = divf %cst, %2 : f64
    %4 = mulf %3, %3 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %5 = mulf %cst_0, %1 : f64
    %6 = divf %5, %4 : f64
    %7 = negf %6 : f64
    %8 = divf %7, %cst_0 : f64
    %9 = mulf %cst_0, %8 : f64
    %10 = addf %cst, %9 : f64
    affine.store %10, %arg0[0] : memref<1xf64>
    return
  }
  func @S8(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0, %arg1] : memref<1000x1000xf64>
    return
  }
  func @S9(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%arg1, %c0] : memref<1000x1000xf64>
    return
  }
  func @S10(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: memref<1000x1000xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg2[%c0, %arg1] : memref<1000x1000xf64>
    affine.store %0, %arg0[%arg1, %c0] : memref<1000x1000xf64>
    return
  }
  func @S11(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1xf64>, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg4[0] : memref<1xf64>
    %2 = affine.apply #map2(%arg2)
    %3 = affine.load %arg0[%arg1, %2] : memref<1000x1000xf64>
    %4 = mulf %1, %3 : f64
    %5 = affine.load %arg3[0] : memref<1xf64>
    %6 = addf %4, %5 : f64
    %7 = divf %0, %6 : f64
    affine.store %7, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func @S12(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1xf64>, %arg6: memref<1000x1000xf64>, %arg7: memref<1xf64>, %arg8: memref<1xf64>, %arg9: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg9[0] : memref<1xf64>
    %1 = affine.apply #map2(%arg1)
    %2 = affine.load %arg6[%arg2, %1] : memref<1000x1000xf64>
    %3 = mulf %0, %2 : f64
    %4 = affine.load %arg8[0] : memref<1xf64>
    %5 = affine.load %arg6[%arg2, %arg1] : memref<1000x1000xf64>
    %6 = mulf %4, %5 : f64
    %7 = addf %3, %6 : f64
    %8 = affine.load %arg7[0] : memref<1xf64>
    %9 = affine.apply #map3(%arg1)
    %10 = affine.load %arg6[%arg2, %9] : memref<1000x1000xf64>
    %11 = mulf %8, %10 : f64
    %12 = subf %7, %11 : f64
    %13 = affine.load %arg5[0] : memref<1xf64>
    %14 = affine.apply #map2(%arg2)
    %15 = affine.load %arg0[%arg1, %14] : memref<1000x1000xf64>
    %16 = mulf %13, %15 : f64
    %17 = subf %12, %16 : f64
    %18 = affine.load %arg4[%arg1, %14] : memref<1000x1000xf64>
    %19 = mulf %13, %18 : f64
    %20 = affine.load %arg3[0] : memref<1xf64>
    %21 = addf %19, %20 : f64
    %22 = divf %17, %21 : f64
    affine.store %22, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func @S13(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>) attributes {scop.stmt} {
    %0 = affine.apply #map4(%arg2)
    %1 = affine.load %arg0[%0, %arg1] : memref<1000x1000xf64>
    %2 = affine.apply #map2(%arg2)
    %3 = affine.load %arg4[%arg1, %2] : memref<1000x1000xf64>
    %4 = mulf %3, %1 : f64
    %5 = affine.load %arg3[%arg1, %2] : memref<1000x1000xf64>
    %6 = addf %4, %5 : f64
    affine.store %6, %arg0[%2, %arg1] : memref<1000x1000xf64>
    return
  }
  func @S14(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%arg1, %c0] : memref<1000x1000xf64>
    return
  }
  func @S15(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%arg1, %c0] : memref<1000x1000xf64>
    return
  }
  func @S16(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: memref<1000x1000xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg2[%arg1, %c0] : memref<1000x1000xf64>
    affine.store %0, %arg0[%arg1, %c0] : memref<1000x1000xf64>
    return
  }
  func @S17(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1xf64>, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg4[0] : memref<1xf64>
    %2 = affine.apply #map2(%arg2)
    %3 = affine.load %arg0[%arg1, %2] : memref<1000x1000xf64>
    %4 = mulf %1, %3 : f64
    %5 = affine.load %arg3[0] : memref<1xf64>
    %6 = addf %4, %5 : f64
    %7 = divf %0, %6 : f64
    affine.store %7, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func @S18(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1xf64>, %arg6: memref<1000x1000xf64>, %arg7: memref<1xf64>, %arg8: memref<1xf64>, %arg9: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg9[0] : memref<1xf64>
    %1 = affine.apply #map2(%arg1)
    %2 = affine.load %arg6[%1, %arg2] : memref<1000x1000xf64>
    %3 = mulf %0, %2 : f64
    %4 = affine.load %arg8[0] : memref<1xf64>
    %5 = affine.load %arg6[%arg1, %arg2] : memref<1000x1000xf64>
    %6 = mulf %4, %5 : f64
    %7 = addf %3, %6 : f64
    %8 = affine.load %arg7[0] : memref<1xf64>
    %9 = affine.apply #map3(%arg1)
    %10 = affine.load %arg6[%9, %arg2] : memref<1000x1000xf64>
    %11 = mulf %8, %10 : f64
    %12 = subf %7, %11 : f64
    %13 = affine.load %arg5[0] : memref<1xf64>
    %14 = affine.apply #map2(%arg2)
    %15 = affine.load %arg0[%arg1, %14] : memref<1000x1000xf64>
    %16 = mulf %13, %15 : f64
    %17 = subf %12, %16 : f64
    %18 = affine.load %arg4[%arg1, %14] : memref<1000x1000xf64>
    %19 = mulf %13, %18 : f64
    %20 = affine.load %arg3[0] : memref<1xf64>
    %21 = addf %19, %20 : f64
    %22 = divf %17, %21 : f64
    affine.store %22, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func @S19(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>) attributes {scop.stmt} {
    %0 = affine.apply #map4(%arg2)
    %1 = affine.load %arg0[%arg1, %0] : memref<1000x1000xf64>
    %2 = affine.apply #map2(%arg2)
    %3 = affine.load %arg4[%arg1, %2] : memref<1000x1000xf64>
    %4 = mulf %3, %1 : f64
    %5 = affine.load %arg3[%arg1, %2] : memref<1000x1000xf64>
    %6 = addf %4, %5 : f64
    affine.store %6, %arg0[%arg1, %2] : memref<1000x1000xf64>
    return
  }
}

