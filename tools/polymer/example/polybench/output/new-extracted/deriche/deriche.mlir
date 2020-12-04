module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2f \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("imgOut\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c0 = constant 0 : index
    %c4096_i32 = constant 4096 : i32
    %c2160_i32 = constant 2160 : i32
    %c0_i32 = constant 0 : i32
    %0 = alloca() : memref<1xf32>
    %1 = alloc() : memref<4096x2160xf32>
    %2 = alloc() : memref<4096x2160xf32>
    %3 = alloc() : memref<4096x2160xf32>
    %4 = alloc() : memref<4096x2160xf32>
    %5 = memref_cast %0 : memref<1xf32> to memref<?xf32>
    call @init_array(%c4096_i32, %c2160_i32, %5, %1, %2) : (i32, i32, memref<?xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    %6 = load %0[%c0] : memref<1xf32>
    call @kernel_deriche(%c4096_i32, %c2160_i32, %6, %1, %2, %3, %4) : (i32, i32, f32, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @print_array(%c4096_i32, %c2160_i32, %2) : (i32, i32, memref<4096x2160xf32>) -> ()
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>) {
    %c0 = constant 0 : index
    %cst = constant 2.500000e-01 : f64
    %c0_i32 = constant 0 : i32
    %c313_i32 = constant 313 : i32
    %c991_i32 = constant 991 : i32
    %c65536_i32 = constant 65536 : i32
    %cst_0 = constant 6.553500e+04 : f32
    %c1_i32 = constant 1 : i32
    %0 = fptrunc %cst : f64 to f32
    store %0, %arg2[%c0] : memref<?xf32>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb5
    %2 = cmpi "slt", %1, %arg0 : i32
    %3 = index_cast %1 : i32 to index
    cond_br %2, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    return
  ^bb3(%4: i32):  // 2 preds: ^bb1, ^bb4
    %5 = cmpi "slt", %4, %arg1 : i32
    %6 = index_cast %4 : i32 to index
    cond_br %5, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %7 = muli %1, %c313_i32 : i32
    %8 = muli %4, %c991_i32 : i32
    %9 = addi %7, %8 : i32
    %10 = remi_signed %9, %c65536_i32 : i32
    %11 = sitofp %10 : i32 to f32
    %12 = divf %11, %cst_0 : f32
    store %12, %arg3[%3, %6] : memref<4096x2160xf32>
    %13 = addi %4, %c1_i32 : i32
    br ^bb3(%13 : i32)
  ^bb5:  // pred: ^bb3
    %14 = addi %1, %c1_i32 : i32
    br ^bb1(%14 : i32)
  }
  func private @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %c1_i32 = constant 1 : i32
    %cst_1 = constant 0.000000e+00 : f32
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
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
    %12 = negf %arg2 : f32
    %13 = exp %12 : f32
    %14 = subf %cst, %13 : f32
    %15 = mulf %14, %14 : f32
    %16 = mulf %cst_0, %arg2 : f32
    %17 = mulf %16, %13 : f32
    %18 = addf %cst, %17 : f32
    %19 = exp %16 : f32
    %20 = subf %18, %19 : f32
    %21 = divf %15, %20 : f32
    %22 = mulf %21, %13 : f32
    %23 = subf %arg2, %cst : f32
    %24 = mulf %22, %23 : f32
    %25 = addf %arg2, %cst : f32
    %26 = mulf %22, %25 : f32
    %27 = negf %21 : f32
    %28 = negf %cst_0 : f32
    %29 = mulf %28, %arg2 : f32
    %30 = exp %29 : f32
    %31 = mulf %27, %30 : f32
    %32 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
    %33 = llvm.mlir.cast %12 : f32 to !llvm.float
    %34 = "llvm.intr.pow"(%32, %33) : (!llvm.float, !llvm.float) -> !llvm.float
    %35 = llvm.mlir.cast %34 : !llvm.float to f32
    %36 = negf %30 : f32
    %37 = sitofp %c1_i32 : i32 to f32
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_1, %4[0] : memref<1xf32>
      affine.store %cst_1, %5[0] : memref<1xf32>
      affine.store %cst_1, %2[0] : memref<1xf32>
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
        %39 = mulf %21, %38 : f32
        %40 = affine.load %2[0] : memref<1xf32>
        %41 = mulf %24, %40 : f32
        %42 = addf %39, %41 : f32
        %43 = affine.load %4[0] : memref<1xf32>
        %44 = mulf %35, %43 : f32
        %45 = addf %42, %44 : f32
        %46 = affine.load %5[0] : memref<1xf32>
        %47 = mulf %36, %46 : f32
        %48 = addf %45, %47 : f32
        affine.store %48, %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        %49 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
        affine.store %49, %2[0] : memref<1xf32>
        %50 = affine.load %4[0] : memref<1xf32>
        affine.store %50, %5[0] : memref<1xf32>
        %51 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        affine.store %51, %4[0] : memref<1xf32>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_1, %10[0] : memref<1xf32>
      affine.store %cst_1, %11[0] : memref<1xf32>
      affine.store %cst_1, %6[0] : memref<1xf32>
      affine.store %cst_1, %7[0] : memref<1xf32>
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %6[0] : memref<1xf32>
        %39 = mulf %26, %38 : f32
        %40 = affine.load %7[0] : memref<1xf32>
        %41 = mulf %31, %40 : f32
        %42 = addf %39, %41 : f32
        %43 = affine.load %10[0] : memref<1xf32>
        %44 = mulf %35, %43 : f32
        %45 = addf %42, %44 : f32
        %46 = affine.load %11[0] : memref<1xf32>
        %47 = mulf %36, %46 : f32
        %48 = addf %45, %47 : f32
        affine.store %48, %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<4096x2160xf32>
        %49 = affine.load %6[0] : memref<1xf32>
        affine.store %49, %7[0] : memref<1xf32>
        %50 = affine.load %arg3[%arg7, -%arg8 + symbol(%0) - 1] : memref<4096x2160xf32>
        affine.store %50, %6[0] : memref<1xf32>
        %51 = affine.load %10[0] : memref<1xf32>
        affine.store %51, %11[0] : memref<1xf32>
        %52 = affine.load %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<4096x2160xf32>
        affine.store %52, %10[0] : memref<1xf32>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        %39 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
        %40 = addf %38, %39 : f32
        %41 = mulf %37, %40 : f32
        affine.store %41, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
      }
    }
    affine.for %arg7 = 0 to %0 {
      affine.store %cst_1, %3[0] : memref<1xf32>
      affine.store %cst_1, %4[0] : memref<1xf32>
      affine.store %cst_1, %5[0] : memref<1xf32>
      affine.for %arg8 = 0 to %1 {
        %38 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
        %39 = mulf %21, %38 : f32
        %40 = affine.load %3[0] : memref<1xf32>
        %41 = mulf %24, %40 : f32
        %42 = addf %39, %41 : f32
        %43 = affine.load %4[0] : memref<1xf32>
        %44 = mulf %35, %43 : f32
        %45 = addf %42, %44 : f32
        %46 = affine.load %5[0] : memref<1xf32>
        %47 = mulf %36, %46 : f32
        %48 = addf %45, %47 : f32
        affine.store %48, %arg5[%arg8, %arg7] : memref<4096x2160xf32>
        %49 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
        affine.store %49, %3[0] : memref<1xf32>
        %50 = affine.load %4[0] : memref<1xf32>
        affine.store %50, %5[0] : memref<1xf32>
        %51 = affine.load %arg5[%arg8, %arg7] : memref<4096x2160xf32>
        affine.store %51, %4[0] : memref<1xf32>
      }
    }
    affine.for %arg7 = 0 to %0 {
      affine.store %cst_1, %8[0] : memref<1xf32>
      affine.store %cst_1, %9[0] : memref<1xf32>
      affine.store %cst_1, %10[0] : memref<1xf32>
      affine.store %cst_1, %11[0] : memref<1xf32>
      affine.for %arg8 = 0 to %1 {
        %38 = affine.load %8[0] : memref<1xf32>
        %39 = mulf %26, %38 : f32
        %40 = affine.load %9[0] : memref<1xf32>
        %41 = mulf %31, %40 : f32
        %42 = addf %39, %41 : f32
        %43 = affine.load %10[0] : memref<1xf32>
        %44 = mulf %35, %43 : f32
        %45 = addf %42, %44 : f32
        %46 = affine.load %11[0] : memref<1xf32>
        %47 = mulf %36, %46 : f32
        %48 = addf %45, %47 : f32
        affine.store %48, %arg6[-%arg8 + symbol(%1) - 1, %arg7] : memref<4096x2160xf32>
        %49 = affine.load %8[0] : memref<1xf32>
        affine.store %49, %9[0] : memref<1xf32>
        %50 = affine.load %arg4[-%arg8 + symbol(%1) - 1, %arg7] : memref<4096x2160xf32>
        affine.store %50, %8[0] : memref<1xf32>
        %51 = affine.load %10[0] : memref<1xf32>
        affine.store %51, %11[0] : memref<1xf32>
        %52 = affine.load %arg6[-%arg8 + symbol(%1) - 1, %arg7] : memref<4096x2160xf32>
        affine.store %52, %10[0] : memref<1xf32>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
        %39 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
        %40 = addf %38, %39 : f32
        %41 = mulf %37, %40 : f32
        affine.store %41, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
      }
    }
    return
  }
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<4096x2160xf32>) {
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.mlir.constant(0 : index) : !llvm.i64
    %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %8 = llvm.mlir.addressof @str1 : !llvm.ptr<array<15 x i8>>
    %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<7 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %16 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %17 = llvm.load %16 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %18 = llvm.mlir.addressof @str5 : !llvm.ptr<array<17 x i8>>
    %19 = llvm.getelementptr %18[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %20 = llvm.mlir.addressof @str2 : !llvm.ptr<array<7 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = llvm.call @fprintf(%17, %19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %23 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %24 = llvm.load %23 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %25 = llvm.mlir.addressof @str6 : !llvm.ptr<array<23 x i8>>
    %26 = llvm.getelementptr %25[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %27 = llvm.call @fprintf(%24, %26) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  ^bb3(%28: i32):  // 2 preds: ^bb1, ^bb4
    %29 = cmpi "slt", %28, %arg1 : i32
    %30 = index_cast %28 : i32 to index
    cond_br %29, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %31 = muli %13, %arg1 : i32
    %32 = addi %31, %28 : i32
    %33 = remi_signed %32, %c20_i32 : i32
    %34 = cmpi "eq", %33, %c0_i32 : i32
    scf.if %34 {
      %45 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %46 = llvm.load %45 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %47 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
      %48 = llvm.getelementptr %47[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %49 = llvm.call @fprintf(%46, %48) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg2[%15, %30] : memref<4096x2160xf32>
    %40 = fpext %39 : f32 to f64
    %41 = llvm.mlir.cast %40 : f64 to !llvm.double
    %42 = llvm.call @fprintf(%36, %38, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %43 = addi %28, %c1_i32 : i32
    br ^bb3(%43 : i32)
  ^bb5:  // pred: ^bb3
    %44 = addi %13, %c1_i32 : i32
    br ^bb1(%44 : i32)
  }
  func private @free(memref<?xi8>)
}

