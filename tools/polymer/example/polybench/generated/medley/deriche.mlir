#map = affine_map<(d0) -> (d0)>
module  {
  func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
    %c0 = constant 0 : index
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %c1_i32 = constant 1 : i32
    %cst_1 = constant 0.000000e+00 : f32
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
    %10 = negf %arg2 : f32
    %11 = exp %10 : f32
    %12 = subf %cst, %11 : f32
    %13 = mulf %12, %12 : f32
    %14 = mulf %cst_0, %arg2 : f32
    %15 = mulf %14, %11 : f32
    %16 = addf %cst, %15 : f32
    %17 = exp %14 : f32
    %18 = subf %16, %17 : f32
    %19 = divf %13, %18 : f32
    %20 = mulf %19, %11 : f32
    %21 = subf %arg2, %cst : f32
    %22 = mulf %20, %21 : f32
    %23 = addf %arg2, %cst : f32
    %24 = mulf %20, %23 : f32
    %25 = negf %19 : f32
    %26 = negf %cst_0 : f32
    %27 = mulf %26, %arg2 : f32
    %28 = exp %27 : f32
    %29 = mulf %25, %28 : f32
    %30 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
    %31 = llvm.mlir.cast %10 : f32 to !llvm.float
    %32 = "llvm.intr.pow"(%30, %31) : (!llvm.float, !llvm.float) -> !llvm.float
    %33 = llvm.mlir.cast %32 : !llvm.float to f32
    %34 = negf %28 : f32
    %35 = sitofp %c1_i32 : i32 to f32
    %36 = index_cast %arg0 : i32 to index
    affine.store %cst_1, %2[%c0] : memref<1xf32>
    affine.store %cst_1, %3[%c0] : memref<1xf32>
    affine.store %cst_1, %0[%c0] : memref<1xf32>
    %37 = index_cast %arg1 : i32 to index
    %38 = affine.load %0[%c0] : memref<1xf32>
    %39 = mulf %22, %38 : f32
    %40 = affine.load %2[%c0] : memref<1xf32>
    %41 = mulf %33, %40 : f32
    %42 = affine.load %3[%c0] : memref<1xf32>
    %43 = mulf %34, %42 : f32
    %44 = affine.load %2[%c0] : memref<1xf32>
    affine.store %44, %3[%c0] : memref<1xf32>
    affine.for %arg7 = 0 to %36 {
      affine.for %arg8 = 0 to %37 {
        %84 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
        %85 = mulf %19, %84 : f32
        %86 = addf %85, %39 : f32
        %87 = addf %86, %41 : f32
        %88 = addf %87, %43 : f32
        affine.store %88, %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        %89 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
        affine.store %89, %0[%c0] : memref<1xf32>
        %90 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        affine.store %90, %2[%c0] : memref<1xf32>
      }
    }
    affine.store %cst_1, %8[%c0] : memref<1xf32>
    affine.store %cst_1, %9[%c0] : memref<1xf32>
    affine.store %cst_1, %4[%c0] : memref<1xf32>
    affine.store %cst_1, %5[%c0] : memref<1xf32>
    %45 = subi %37, %c1 : index
    %46 = addi %45, %c1 : index
    %47 = subi %46, %c1 : index
    %48 = affine.load %4[%c0] : memref<1xf32>
    %49 = mulf %24, %48 : f32
    %50 = affine.load %5[%c0] : memref<1xf32>
    %51 = mulf %29, %50 : f32
    %52 = addf %49, %51 : f32
    %53 = affine.load %8[%c0] : memref<1xf32>
    %54 = mulf %33, %53 : f32
    %55 = addf %52, %54 : f32
    %56 = affine.load %9[%c0] : memref<1xf32>
    %57 = mulf %34, %56 : f32
    %58 = addf %55, %57 : f32
    %59 = affine.load %4[%c0] : memref<1xf32>
    affine.store %59, %5[%c0] : memref<1xf32>
    %60 = affine.load %8[%c0] : memref<1xf32>
    affine.store %60, %9[%c0] : memref<1xf32>
    affine.for %arg7 = 0 to %36 {
      affine.for %arg8 = 0 to %37 {
        %84 = affine.apply #map(%arg8)
        affine.store %58, %arg6[%arg7, %84] : memref<4096x2160xf32>
        %85 = affine.load %arg3[%arg7, %84] : memref<4096x2160xf32>
        affine.store %85, %4[%c0] : memref<1xf32>
        %86 = affine.load %arg6[%arg7, %84] : memref<4096x2160xf32>
        affine.store %86, %8[%c0] : memref<1xf32>
      }
    }
    affine.for %arg7 = 0 to %36 {
      affine.for %arg8 = 0 to %37 {
        %84 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        %85 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
        %86 = addf %84, %85 : f32
        %87 = mulf %35, %86 : f32
        affine.store %87, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
      }
    }
    affine.store %cst_1, %1[%c0] : memref<1xf32>
    affine.store %cst_1, %2[%c0] : memref<1xf32>
    affine.store %cst_1, %3[%c0] : memref<1xf32>
    %61 = affine.load %1[%c0] : memref<1xf32>
    %62 = mulf %22, %61 : f32
    %63 = affine.load %2[%c0] : memref<1xf32>
    %64 = mulf %33, %63 : f32
    %65 = affine.load %3[%c0] : memref<1xf32>
    %66 = mulf %34, %65 : f32
    %67 = affine.load %2[%c0] : memref<1xf32>
    affine.store %67, %3[%c0] : memref<1xf32>
    affine.for %arg7 = 0 to %37 {
      affine.for %arg8 = 0 to %36 {
        %84 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
        %85 = mulf %19, %84 : f32
        %86 = addf %85, %62 : f32
        %87 = addf %86, %64 : f32
        %88 = addf %87, %66 : f32
        affine.store %88, %arg5[%arg8, %arg7] : memref<4096x2160xf32>
        %89 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
        affine.store %89, %1[%c0] : memref<1xf32>
        %90 = affine.load %arg5[%arg8, %arg7] : memref<4096x2160xf32>
        affine.store %90, %2[%c0] : memref<1xf32>
      }
    }
    affine.store %cst_1, %6[%c0] : memref<1xf32>
    affine.store %cst_1, %7[%c0] : memref<1xf32>
    affine.store %cst_1, %8[%c0] : memref<1xf32>
    affine.store %cst_1, %9[%c0] : memref<1xf32>
    %68 = subi %36, %c1 : index
    %69 = addi %68, %c1 : index
    %70 = subi %69, %c1 : index
    %71 = affine.load %6[%c0] : memref<1xf32>
    %72 = mulf %24, %71 : f32
    %73 = affine.load %7[%c0] : memref<1xf32>
    %74 = mulf %29, %73 : f32
    %75 = addf %72, %74 : f32
    %76 = affine.load %8[%c0] : memref<1xf32>
    %77 = mulf %33, %76 : f32
    %78 = addf %75, %77 : f32
    %79 = affine.load %9[%c0] : memref<1xf32>
    %80 = mulf %34, %79 : f32
    %81 = addf %78, %80 : f32
    %82 = affine.load %6[%c0] : memref<1xf32>
    affine.store %82, %7[%c0] : memref<1xf32>
    %83 = affine.load %8[%c0] : memref<1xf32>
    affine.store %83, %9[%c0] : memref<1xf32>
    affine.for %arg7 = 0 to %37 {
      affine.for %arg8 = 0 to %36 {
        %84 = affine.apply #map(%arg8)
        affine.store %81, %arg6[%84, %arg7] : memref<4096x2160xf32>
        %85 = affine.load %arg4[%84, %arg7] : memref<4096x2160xf32>
        affine.store %85, %6[%c0] : memref<1xf32>
        %86 = affine.load %arg6[%84, %arg7] : memref<4096x2160xf32>
        affine.store %86, %8[%c0] : memref<1xf32>
      }
    }
    affine.for %arg7 = 0 to %36 {
      affine.for %arg8 = 0 to %37 {
        %84 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        %85 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
        %86 = addf %84, %85 : f32
        %87 = mulf %35, %86 : f32
        affine.store %87, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
      }
    }
    return
  }
}
