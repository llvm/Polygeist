#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0 * 32)>
#map5 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map6 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>


module {
  func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
    %c1 = constant 1 : index
    %0 = alloca() : memref<1xf32>
    %1 = alloca() : memref<1xf32>
    %2 = alloca() : memref<1xf32>
    %3 = alloca() : memref<1xf32>
    %4 = alloca() : memref<1xf32>
    %5 = alloca() : memref<1xf32>
    %6 = alloca() : memref<1xf32>
    %7 = alloca() : memref<1xf32>
    %8 = alloca() : memref<1xf32>
    %9 = alloca() : memref<1xf32>
    %10 = alloca() : memref<1xf32>
    call @S0(%10, %arg2) : (memref<1xf32>, f32) -> ()
    %11 = alloca() : memref<1xf32>
    call @S1(%11) : (memref<1xf32>) -> ()
    %12 = index_cast %arg0 : i32 to index
    call @S2(%2) : (memref<1xf32>) -> ()
    call @S3(%3) : (memref<1xf32>) -> ()
    call @S4(%0) : (memref<1xf32>) -> ()
    %13 = index_cast %arg1 : i32 to index
    %14 = alloca() : memref<1xf32>
    call @S5(%14, %0, %arg2) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    %15 = alloca() : memref<1xf32>
    call @S6(%15, %2, %arg2) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    %16 = alloca() : memref<1xf32>
    call @S7(%16, %3, %arg2) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S8(%3, %2) : (memref<1xf32>, memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %12 {
      affine.for %arg8 = 0 to %13 {
        call @S9(%arg5, %arg7, %arg8, %16, %15, %14, %arg3, %10) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S10(%0, %arg3, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S11(%2, %arg5, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S12(%8) : (memref<1xf32>) -> ()
    call @S13(%9) : (memref<1xf32>) -> ()
    call @S14(%4) : (memref<1xf32>) -> ()
    call @S15(%5) : (memref<1xf32>) -> ()
    %17 = subi %13, %c1 : index
    %18 = addi %17, %c1 : index
    %19 = alloca() : memref<1xf32>
    call @S16(%19, %9, %arg2, %8, %5, %4) : (memref<1xf32>, memref<1xf32>, f32, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
    call @S17(%5, %4) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S18(%9, %8) : (memref<1xf32>, memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %12 {
      affine.for %arg8 = 0 to %13 {
        call @S19(%arg6, %arg7, %arg8, %19) : (memref<4096x2160xf32>, index, index, memref<1xf32>) -> ()
        call @S20(%4, %arg3, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S21(%8, %arg6, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    affine.for %arg7 = 0 to %12 {
      affine.for %arg8 = 0 to %13 {
        call @S22(%arg4, %arg7, %arg8, %arg6, %arg5, %11) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
      }
    }
    call @S23(%1) : (memref<1xf32>) -> ()
    call @S24(%2) : (memref<1xf32>) -> ()
    call @S25(%3) : (memref<1xf32>) -> ()
    %20 = alloca() : memref<1xf32>
    call @S26(%20, %1, %arg2) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    %21 = alloca() : memref<1xf32>
    call @S27(%21, %2, %arg2) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    %22 = alloca() : memref<1xf32>
    call @S28(%22, %3, %arg2) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S29(%3, %2) : (memref<1xf32>, memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %13 {
      affine.for %arg8 = 0 to %12 {
        call @S30(%arg5, %arg8, %arg7, %22, %21, %20, %arg4, %10) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S31(%1, %arg4, %arg8, %arg7) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S32(%2, %arg5, %arg8, %arg7) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S33(%6) : (memref<1xf32>) -> ()
    call @S34(%7) : (memref<1xf32>) -> ()
    call @S35(%8) : (memref<1xf32>) -> ()
    call @S36(%9) : (memref<1xf32>) -> ()
    %23 = subi %12, %c1 : index
    %24 = addi %23, %c1 : index
    %25 = alloca() : memref<1xf32>
    call @S37(%25, %9, %arg2, %8, %7, %6) : (memref<1xf32>, memref<1xf32>, f32, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
    call @S38(%7, %6) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S39(%9, %8) : (memref<1xf32>, memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %13 {
      affine.for %arg8 = 0 to %12 {
        call @S40(%arg6, %arg7, %arg8, %25) : (memref<4096x2160xf32>, index, index, memref<1xf32>) -> ()
        call @S41(%6, %arg4, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S42(%8, %arg6, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    affine.for %arg7 = 0 to %12 {
      affine.for %arg8 = 0 to %13 {
        call @S43(%arg4, %arg7, %arg8, %arg6, %arg5, %11) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f32
    %0 = negf %arg1 : f32
    %1 = exp %0 : f32
    %2 = subf %cst, %1 : f32
    %3 = mulf %2, %2 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %4 = mulf %cst_0, %arg1 : f32
    %5 = mulf %4, %1 : f32
    %6 = addf %cst, %5 : f32
    %7 = exp %4 : f32
    %8 = subf %6, %7 : f32
    %9 = divf %3, %8 : f32
    affine.store %9, %arg0[0] : memref<1xf32>
    return
  }
  func @S1(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = sitofp %c1_i32 : i32 to f32
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func @S2(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S3(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S4(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S5(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = mulf %cst, %arg2 : f32
    %1 = exp %0 : f32
    %2 = negf %arg2 : f32
    %3 = exp %2 : f32
    %4 = mulf %0, %3 : f32
    %cst_0 = constant 1.000000e+00 : f32
    %5 = subf %cst_0, %3 : f32
    %6 = mulf %5, %5 : f32
    %7 = addf %cst_0, %4 : f32
    %8 = subf %7, %1 : f32
    %9 = divf %6, %8 : f32
    %10 = mulf %9, %3 : f32
    %11 = subf %arg2, %cst_0 : f32
    %12 = mulf %10, %11 : f32
    %c0 = constant 0 : index
    %13 = affine.load %arg1[%c0] : memref<1xf32>
    %14 = mulf %12, %13 : f32
    affine.store %14, %arg0[0] : memref<1xf32>
    return
  }
  func @S6(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = llvm.mlir.cast %cst : f32 to !llvm.float
    %1 = negf %arg2 : f32
    %2 = llvm.mlir.cast %1 : f32 to !llvm.float
    %3 = "llvm.intr.pow"(%0, %2) : (!llvm.float, !llvm.float) -> !llvm.float
    %4 = llvm.mlir.cast %3 : !llvm.float to f32
    %c0 = constant 0 : index
    %5 = affine.load %arg1[%c0] : memref<1xf32>
    %6 = mulf %4, %5 : f32
    affine.store %6, %arg0[0] : memref<1xf32>
    return
  }
  func @S7(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = negf %cst : f32
    %1 = mulf %0, %arg2 : f32
    %2 = exp %1 : f32
    %3 = negf %2 : f32
    %c0 = constant 0 : index
    %4 = affine.load %arg1[%c0] : memref<1xf32>
    %5 = mulf %3, %4 : f32
    affine.store %5, %arg0[0] : memref<1xf32>
    return
  }
  func @S8(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf32>
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S9(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<1xf32>, %arg4: memref<1xf32>, %arg5: memref<1xf32>, %arg6: memref<4096x2160xf32>, %arg7: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg7[0] : memref<1xf32>
    %1 = affine.load %arg6[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %arg5[0] : memref<1xf32>
    %4 = addf %2, %3 : f32
    %5 = affine.load %arg4[0] : memref<1xf32>
    %6 = addf %4, %5 : f32
    %7 = affine.load %arg3[0] : memref<1xf32>
    %8 = addf %6, %7 : f32
    affine.store %8, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func @S10(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S11(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S12(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S13(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S14(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S15(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S16(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32, %arg3: memref<1xf32>, %arg4: memref<1xf32>, %arg5: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f32
    %0 = addf %arg2, %cst : f32
    %1 = negf %arg2 : f32
    %2 = exp %1 : f32
    %3 = subf %cst, %2 : f32
    %4 = mulf %3, %3 : f32
    %5 = llvm.mlir.cast %1 : f32 to !llvm.float
    %cst_0 = constant 2.000000e+00 : f32
    %6 = mulf %cst_0, %arg2 : f32
    %7 = mulf %6, %2 : f32
    %8 = addf %cst, %7 : f32
    %9 = exp %6 : f32
    %10 = subf %8, %9 : f32
    %11 = divf %4, %10 : f32
    %12 = mulf %11, %2 : f32
    %13 = mulf %12, %0 : f32
    %14 = negf %11 : f32
    %15 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
    %16 = "llvm.intr.pow"(%15, %5) : (!llvm.float, !llvm.float) -> !llvm.float
    %17 = llvm.mlir.cast %16 : !llvm.float to f32
    %18 = negf %cst_0 : f32
    %19 = mulf %18, %arg2 : f32
    %20 = exp %19 : f32
    %21 = mulf %14, %20 : f32
    %22 = negf %20 : f32
    %c0 = constant 0 : index
    %23 = affine.load %arg5[%c0] : memref<1xf32>
    %24 = mulf %13, %23 : f32
    %25 = affine.load %arg4[%c0] : memref<1xf32>
    %26 = mulf %21, %25 : f32
    %27 = addf %24, %26 : f32
    %28 = affine.load %arg3[%c0] : memref<1xf32>
    %29 = mulf %17, %28 : f32
    %30 = addf %27, %29 : f32
    %31 = affine.load %arg1[%c0] : memref<1xf32>
    %32 = mulf %22, %31 : f32
    %33 = addf %30, %32 : f32
    affine.store %33, %arg0[0] : memref<1xf32>
    return
  }
  func @S17(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf32>
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S18(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf32>
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S19(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf32>
    %1 = affine.apply #map2(%arg2)
    affine.store %0, %arg0[%arg1, %1] : memref<4096x2160xf32>
    return
  }
  func @S20(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map2(%arg3)
    %1 = affine.load %arg1[%arg2, %0] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %1, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S21(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map2(%arg3)
    %1 = affine.load %arg1[%arg2, %0] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %1, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S22(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf32>
    %1 = affine.load %arg4[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = affine.load %arg3[%arg1, %arg2] : memref<4096x2160xf32>
    %3 = addf %1, %2 : f32
    %4 = mulf %0, %3 : f32
    affine.store %4, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func @S23(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S24(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S25(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S26(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = mulf %cst, %arg2 : f32
    %1 = exp %0 : f32
    %2 = negf %arg2 : f32
    %3 = exp %2 : f32
    %4 = mulf %0, %3 : f32
    %cst_0 = constant 1.000000e+00 : f32
    %5 = subf %cst_0, %3 : f32
    %6 = mulf %5, %5 : f32
    %7 = addf %cst_0, %4 : f32
    %8 = subf %7, %1 : f32
    %9 = divf %6, %8 : f32
    %10 = mulf %9, %3 : f32
    %11 = subf %arg2, %cst_0 : f32
    %12 = mulf %10, %11 : f32
    %c0 = constant 0 : index
    %13 = affine.load %arg1[%c0] : memref<1xf32>
    %14 = mulf %12, %13 : f32
    affine.store %14, %arg0[0] : memref<1xf32>
    return
  }
  func @S27(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = llvm.mlir.cast %cst : f32 to !llvm.float
    %1 = negf %arg2 : f32
    %2 = llvm.mlir.cast %1 : f32 to !llvm.float
    %3 = "llvm.intr.pow"(%0, %2) : (!llvm.float, !llvm.float) -> !llvm.float
    %4 = llvm.mlir.cast %3 : !llvm.float to f32
    %c0 = constant 0 : index
    %5 = affine.load %arg1[%c0] : memref<1xf32>
    %6 = mulf %4, %5 : f32
    affine.store %6, %arg0[0] : memref<1xf32>
    return
  }
  func @S28(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = negf %cst : f32
    %1 = mulf %0, %arg2 : f32
    %2 = exp %1 : f32
    %3 = negf %2 : f32
    %c0 = constant 0 : index
    %4 = affine.load %arg1[%c0] : memref<1xf32>
    %5 = mulf %3, %4 : f32
    affine.store %5, %arg0[0] : memref<1xf32>
    return
  }
  func @S29(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf32>
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S30(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<1xf32>, %arg4: memref<1xf32>, %arg5: memref<1xf32>, %arg6: memref<4096x2160xf32>, %arg7: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg7[0] : memref<1xf32>
    %1 = affine.load %arg6[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %arg5[0] : memref<1xf32>
    %4 = addf %2, %3 : f32
    %5 = affine.load %arg4[0] : memref<1xf32>
    %6 = addf %4, %5 : f32
    %7 = affine.load %arg3[0] : memref<1xf32>
    %8 = addf %6, %7 : f32
    affine.store %8, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func @S31(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S32(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S33(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S34(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S35(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S36(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S37(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: f32, %arg3: memref<1xf32>, %arg4: memref<1xf32>, %arg5: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f32
    %0 = addf %arg2, %cst : f32
    %1 = negf %arg2 : f32
    %2 = exp %1 : f32
    %3 = subf %cst, %2 : f32
    %4 = mulf %3, %3 : f32
    %5 = llvm.mlir.cast %1 : f32 to !llvm.float
    %cst_0 = constant 2.000000e+00 : f32
    %6 = mulf %cst_0, %arg2 : f32
    %7 = mulf %6, %2 : f32
    %8 = addf %cst, %7 : f32
    %9 = exp %6 : f32
    %10 = subf %8, %9 : f32
    %11 = divf %4, %10 : f32
    %12 = mulf %11, %2 : f32
    %13 = mulf %12, %0 : f32
    %14 = negf %11 : f32
    %15 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
    %16 = "llvm.intr.pow"(%15, %5) : (!llvm.float, !llvm.float) -> !llvm.float
    %17 = llvm.mlir.cast %16 : !llvm.float to f32
    %18 = negf %cst_0 : f32
    %19 = mulf %18, %arg2 : f32
    %20 = exp %19 : f32
    %21 = mulf %14, %20 : f32
    %22 = negf %20 : f32
    %c0 = constant 0 : index
    %23 = affine.load %arg5[%c0] : memref<1xf32>
    %24 = mulf %13, %23 : f32
    %25 = affine.load %arg4[%c0] : memref<1xf32>
    %26 = mulf %21, %25 : f32
    %27 = addf %24, %26 : f32
    %28 = affine.load %arg3[%c0] : memref<1xf32>
    %29 = mulf %17, %28 : f32
    %30 = addf %27, %29 : f32
    %31 = affine.load %arg1[%c0] : memref<1xf32>
    %32 = mulf %22, %31 : f32
    %33 = addf %30, %32 : f32
    affine.store %33, %arg0[0] : memref<1xf32>
    return
  }
  func @S38(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf32>
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S39(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf32>
    affine.store %0, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S40(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf32>
    %1 = affine.apply #map2(%arg2)
    affine.store %0, %arg0[%1, %arg1] : memref<4096x2160xf32>
    return
  }
  func @S41(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map2(%arg3)
    %1 = affine.load %arg1[%0, %arg2] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %1, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S42(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map2(%arg3)
    %1 = affine.load %arg1[%0, %arg2] : memref<4096x2160xf32>
    %c0 = constant 0 : index
    affine.store %1, %arg0[%c0] : memref<1xf32>
    return
  }
  func @S43(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf32>
    %1 = affine.load %arg4[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = affine.load %arg3[%arg1, %arg2] : memref<4096x2160xf32>
    %3 = addf %1, %2 : f32
    %4 = mulf %0, %3 : f32
    affine.store %4, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func @kernel_deriche_new(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>, %arg4: f32, %arg5: i32, %arg6: i32) {
    %0 = alloca() : memref<1xf32>
    %1 = alloca() : memref<1xf32>
    %2 = alloca() : memref<1xf32>
    %3 = alloca() : memref<1xf32>
    %4 = alloca() : memref<1xf32>
    %5 = alloca() : memref<1xf32>
    %6 = alloca() : memref<1xf32>
    %7 = alloca() : memref<1xf32>
    %8 = alloca() : memref<1xf32>
    %9 = alloca() : memref<1xf32>
    %10 = alloca() : memref<1xf32>
    %11 = alloca() : memref<1xf32>
    %12 = alloca() : memref<1xf32>
    %13 = alloca() : memref<1xf32>
    %14 = alloca() : memref<1xf32>
    %15 = alloca() : memref<1xf32>
    %16 = alloca() : memref<1xf32>
    %17 = alloca() : memref<1xf32>
    %18 = alloca() : memref<1xf32>
    %19 = alloca() : memref<1xf32>
    %20 = index_cast %arg6 : i32 to index
    %21 = index_cast %arg5 : i32 to index
    %22 = alloca() : memref<1xf32>
    %23 = alloca() : memref<1xf32>
    %24 = alloca() : memref<1xf32>
    %25 = alloca() : memref<1xf32>
    %26 = alloca() : memref<1xf32>
    %27 = alloca() : memref<1xf32>
    %28 = alloca() : memref<1xf32>
    %29 = alloca() : memref<1xf32>
    %30 = alloca() : memref<1xf32>
    %31 = alloca() : memref<1xf32>
    %32 = alloca() : memref<1xf32>
    %33 = alloca() : memref<1xf32>
    %34 = alloca() : memref<1xf32>
    %35 = alloca() : memref<1xf32>
    %36 = alloca() : memref<1xf32>
    call @S3(%19) : (memref<1xf32>) -> ()
    call @S15(%33) : (memref<1xf32>) -> ()
    call @S14(%36) : (memref<1xf32>) -> ()
    call @S13(%35) : (memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %20 {
      affine.for %arg8 = 0 to %21 {
        %37 = alloca() : memref<1xf32>
        %38 = alloca() : memref<1xf32>
        %39 = affine.apply #map2(%arg7)
        %40 = affine.apply #map2(%arg8)
        call @S11(%38, %arg0, %39, %40) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S10(%37, %arg3, %39, %40) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S4(%14) : (memref<1xf32>) -> ()
    affine.for %arg7 = 0 to #map6()[%20] {
      affine.for %arg8 = 0 to #map6()[%21] {
        affine.for %arg9 = #map4(%arg7) to min #map5(%arg7)[%20] {
          affine.for %arg10 = #map4(%arg8) to min #map5(%arg8)[%21] {
            %37 = alloca() : memref<1xf32>
            %38 = alloca() : memref<1xf32>
            %39 = alloca() : memref<1xf32>
            %40 = alloca() : memref<1xf32>
            %41 = alloca() : memref<1xf32>
            %42 = affine.apply #map2(%arg9)
            %43 = affine.apply #map2(%arg10)
            call @S43(%arg2, %42, %43, %arg1, %arg0, %41) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
            call @S9(%arg0, %42, %43, %40, %39, %38, %arg3, %37) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
          }
        }
      }
    }
    call @S7(%13, %19, %arg4) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S5(%12, %14, %arg4) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    affine.for %arg7 = 0 to %20 {
      affine.for %arg8 = 0 to %21 {
        %37 = alloca() : memref<1xf32>
        %38 = affine.apply #map2(%arg7)
        %39 = affine.apply #map2(%arg8)
        call @S32(%37, %arg0, %38, %39) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S2(%11) : (memref<1xf32>) -> ()
    call @S8(%19, %11) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S25(%19) : (memref<1xf32>) -> ()
    call @S6(%10, %11, %arg4) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S24(%11) : (memref<1xf32>) -> ()
    call @S1(%9) : (memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %20 {
      affine.for %arg8 = 0 to %21 {
        %37 = alloca() : memref<1xf32>
        %38 = affine.apply #map2(%arg7)
        %39 = affine.apply #map2(%arg8)
        call @S31(%37, %arg2, %38, %39) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S23(%8) : (memref<1xf32>) -> ()
    affine.for %arg7 = 0 to #map6()[%20] {
      affine.for %arg8 = 0 to #map6()[%21] {
        affine.for %arg9 = #map4(%arg7) to min #map5(%arg7)[%20] {
          affine.for %arg10 = #map4(%arg8) to min #map5(%arg8)[%21] {
            %37 = alloca() : memref<1xf32>
            %38 = alloca() : memref<1xf32>
            %39 = alloca() : memref<1xf32>
            %40 = alloca() : memref<1xf32>
            %41 = affine.apply #map2(%arg9)
            %42 = affine.apply #map2(%arg10)
            call @S30(%arg0, %41, %42, %40, %39, %38, %arg2, %37) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
          }
        }
      }
    }
    call @S28(%7, %19, %arg4) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S29(%19, %11) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S27(%6, %11, %arg4) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S26(%5, %8, %arg4) : (memref<1xf32>, memref<1xf32>, f32) -> ()
    call @S0(%4, %arg4) : (memref<1xf32>, f32) -> ()
    affine.for %arg7 = 0 to %20 {
      affine.for %arg8 = 0 to %21 {
        %37 = alloca() : memref<1xf32>
        %38 = affine.apply #map2(%arg7)
        %39 = affine.apply #map2(%arg8)
        call @S41(%37, %arg2, %38, %39) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S33(%3) : (memref<1xf32>) -> ()
    call @S38(%15, %3) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S34(%15) : (memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %20 {
      affine.for %arg8 = 0 to %21 {
        %37 = alloca() : memref<1xf32>
        %38 = alloca() : memref<1xf32>
        %39 = affine.apply #map2(%arg7)
        %40 = affine.apply #map2(%arg8)
        call @S40(%arg1, %39, %40, %38) : (memref<4096x2160xf32>, index, index, memref<1xf32>) -> ()
        call @S42(%37, %arg1, %39, %40) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S12(%2) : (memref<1xf32>) -> ()
    call @S16(%34, %35, %arg4, %2, %33, %36) : (memref<1xf32>, memref<1xf32>, f32, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
    call @S18(%35, %2) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S36(%35) : (memref<1xf32>) -> ()
    call @S17(%33, %36) : (memref<1xf32>, memref<1xf32>) -> ()
    affine.for %arg7 = 0 to %20 {
      affine.for %arg8 = 0 to %21 {
        %37 = alloca() : memref<1xf32>
        %38 = alloca() : memref<1xf32>
        %39 = alloca() : memref<1xf32>
        %40 = alloca() : memref<1xf32>
        %41 = affine.apply #map2(%arg7)
        %42 = affine.apply #map2(%arg8)
        call @S20(%40, %arg3, %41, %42) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S19(%arg1, %41, %42, %39) : (memref<4096x2160xf32>, index, index, memref<1xf32>) -> ()
        call @S22(%arg2, %41, %42, %arg1, %arg0, %38) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S21(%37, %arg1, %41, %42) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    call @S39(%35, %1) : (memref<1xf32>, memref<1xf32>) -> ()
    call @S35(%1) : (memref<1xf32>) -> ()
    call @S37(%0, %35, %arg4, %1, %15, %3) : (memref<1xf32>, memref<1xf32>, f32, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
    return
  }
}
