#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2lf \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("x\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c40_i32 = constant 40 : i32
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<40x40xf64>
    %1 = alloc() : memref<40xf64>
    %2 = alloc() : memref<40xf64>
    %3 = alloc() : memref<40xf64>
    call @init_array(%c40_i32, %0, %1, %2, %3) : (i32, memref<40x40xf64>, memref<40xf64>, memref<40xf64>, memref<40xf64>) -> ()
    call @kernel_ludcmp(%c40_i32, %0, %1, %2, %3) : (i32, memref<40x40xf64>, memref<40xf64>, memref<40xf64>, memref<40xf64>) -> ()
    call @print_array(%c40_i32, %2) : (i32, memref<40xf64>) -> ()
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: memref<40x40xf64>, %arg2: memref<40xf64>, %arg3: memref<40xf64>, %arg4: memref<40xf64>) {
    %c0_i32 = constant 0 : i32
    %cst = constant 2.000000e+00 : f64
    %c4_i32 = constant 4 : i32
    %c1_i32 = constant 1 : i32
    %0 = sitofp %arg0 : i32 to f64
    br ^bb1(%c0_i32 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = cmpi "slt", %1, %arg0 : i32
    %3 = index_cast %1 : i32 to index
    cond_br %2, ^bb2, ^bb3(%c0_i32 : i32)
  ^bb2:  // pred: ^bb1
    %4 = sitofp %c0_i32 : i32 to f64
    store %4, %arg3[%3] : memref<40xf64>
    store %4, %arg4[%3] : memref<40xf64>
    %5 = addi %1, %c1_i32 : i32
    %6 = sitofp %5 : i32 to f64
    %7 = divf %6, %0 : f64
    %8 = divf %7, %cst : f64
    %9 = sitofp %c4_i32 : i32 to f64
    %10 = addf %8, %9 : f64
    store %10, %arg2[%3] : memref<40xf64>
    br ^bb1(%5 : i32)
  ^bb3(%11: i32):  // 2 preds: ^bb1, ^bb10
    %12 = cmpi "slt", %11, %arg0 : i32
    %13 = index_cast %11 : i32 to index
    cond_br %12, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %14 = alloc() : memref<40x40xf64>
    br ^bb11(%c0_i32 : i32)
  ^bb5(%15: i32):  // 2 preds: ^bb3, ^bb6
    %16 = cmpi "sle", %15, %11 : i32
    %17 = index_cast %15 : i32 to index
    cond_br %16, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %18 = subi %c0_i32, %15 : i32
    %19 = remi_signed %18, %arg0 : i32
    %20 = sitofp %19 : i32 to f64
    %21 = divf %20, %0 : f64
    %22 = sitofp %c1_i32 : i32 to f64
    %23 = addf %21, %22 : f64
    store %23, %arg1[%13, %17] : memref<40x40xf64>
    %24 = addi %15, %c1_i32 : i32
    br ^bb5(%24 : i32)
  ^bb7:  // pred: ^bb5
    %25 = addi %11, %c1_i32 : i32
    br ^bb8(%25 : i32)
  ^bb8(%26: i32):  // 2 preds: ^bb7, ^bb9
    %27 = cmpi "slt", %26, %arg0 : i32
    %28 = index_cast %26 : i32 to index
    cond_br %27, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %29 = sitofp %c0_i32 : i32 to f64
    store %29, %arg1[%13, %28] : memref<40x40xf64>
    %30 = addi %26, %c1_i32 : i32
    br ^bb8(%30 : i32)
  ^bb10:  // pred: ^bb8
    %31 = sitofp %c1_i32 : i32 to f64
    store %31, %arg1[%13, %13] : memref<40x40xf64>
    br ^bb3(%25 : i32)
  ^bb11(%32: i32):  // 2 preds: ^bb4, ^bb14
    %33 = cmpi "slt", %32, %arg0 : i32
    %34 = index_cast %32 : i32 to index
    cond_br %33, ^bb12(%c0_i32 : i32), ^bb15(%c0_i32 : i32)
  ^bb12(%35: i32):  // 2 preds: ^bb11, ^bb13
    %36 = cmpi "slt", %35, %arg0 : i32
    %37 = index_cast %35 : i32 to index
    cond_br %36, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %38 = sitofp %c0_i32 : i32 to f64
    store %38, %14[%34, %37] : memref<40x40xf64>
    %39 = addi %35, %c1_i32 : i32
    br ^bb12(%39 : i32)
  ^bb14:  // pred: ^bb12
    %40 = addi %32, %c1_i32 : i32
    br ^bb11(%40 : i32)
  ^bb15(%41: i32):  // 2 preds: ^bb11, ^bb17
    %42 = cmpi "slt", %41, %arg0 : i32
    %43 = index_cast %41 : i32 to index
    cond_br %42, ^bb16(%c0_i32 : i32), ^bb21(%c0_i32 : i32)
  ^bb16(%44: i32):  // 2 preds: ^bb15, ^bb20
    %45 = cmpi "slt", %44, %arg0 : i32
    %46 = index_cast %44 : i32 to index
    cond_br %45, ^bb18(%c0_i32 : i32), ^bb17
  ^bb17:  // pred: ^bb16
    %47 = addi %41, %c1_i32 : i32
    br ^bb15(%47 : i32)
  ^bb18(%48: i32):  // 2 preds: ^bb16, ^bb19
    %49 = cmpi "slt", %48, %arg0 : i32
    %50 = index_cast %48 : i32 to index
    cond_br %49, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %51 = load %arg1[%46, %43] : memref<40x40xf64>
    %52 = load %arg1[%50, %43] : memref<40x40xf64>
    %53 = mulf %51, %52 : f64
    %54 = load %14[%46, %50] : memref<40x40xf64>
    %55 = addf %54, %53 : f64
    store %55, %14[%46, %50] : memref<40x40xf64>
    %56 = addi %48, %c1_i32 : i32
    br ^bb18(%56 : i32)
  ^bb20:  // pred: ^bb18
    %57 = addi %44, %c1_i32 : i32
    br ^bb16(%57 : i32)
  ^bb21(%58: i32):  // 2 preds: ^bb15, ^bb25
    %59 = cmpi "slt", %58, %arg0 : i32
    %60 = index_cast %58 : i32 to index
    cond_br %59, ^bb23(%c0_i32 : i32), ^bb22
  ^bb22:  // pred: ^bb21
    return
  ^bb23(%61: i32):  // 2 preds: ^bb21, ^bb24
    %62 = cmpi "slt", %61, %arg0 : i32
    %63 = index_cast %61 : i32 to index
    cond_br %62, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %64 = load %14[%60, %63] : memref<40x40xf64>
    store %64, %arg1[%60, %63] : memref<40x40xf64>
    %65 = addi %61, %c1_i32 : i32
    br ^bb23(%65 : i32)
  ^bb25:  // pred: ^bb23
    %66 = addi %58, %c1_i32 : i32
    br ^bb21(%66 : i32)
  }
  func private @kernel_ludcmp(%arg0: i32, %arg1: memref<40x40xf64>, %arg2: memref<40xf64>, %arg3: memref<40xf64>, %arg4: memref<40xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = alloca() : memref<1xf64>
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to #map0(%arg5) {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<40x40xf64>
        affine.store %2, %1[0] : memref<1xf64>
        affine.for %arg7 = 0 to #map0(%arg6) {
          %6 = affine.load %arg1[%arg5, %arg7] : memref<40x40xf64>
          %7 = affine.load %arg1[%arg7, %arg6] : memref<40x40xf64>
          %8 = mulf %6, %7 : f64
          %9 = affine.load %1[0] : memref<1xf64>
          %10 = subf %9, %8 : f64
          affine.store %10, %1[0] : memref<1xf64>
        }
        %3 = affine.load %1[0] : memref<1xf64>
        %4 = affine.load %arg1[%arg6, %arg6] : memref<40x40xf64>
        %5 = divf %3, %4 : f64
        affine.store %5, %arg1[%arg5, %arg6] : memref<40x40xf64>
      }
      affine.for %arg6 = #map0(%arg5) to %0 {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<40x40xf64>
        affine.store %2, %1[0] : memref<1xf64>
        affine.for %arg7 = 0 to #map0(%arg5) {
          %4 = affine.load %arg1[%arg5, %arg7] : memref<40x40xf64>
          %5 = affine.load %arg1[%arg7, %arg6] : memref<40x40xf64>
          %6 = mulf %4, %5 : f64
          %7 = affine.load %1[0] : memref<1xf64>
          %8 = subf %7, %6 : f64
          affine.store %8, %1[0] : memref<1xf64>
        }
        %3 = affine.load %1[0] : memref<1xf64>
        affine.store %3, %arg1[%arg5, %arg6] : memref<40x40xf64>
      }
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg2[%arg5] : memref<40xf64>
      affine.store %2, %1[0] : memref<1xf64>
      affine.for %arg6 = 0 to #map0(%arg5) {
        %4 = affine.load %arg1[%arg5, %arg6] : memref<40x40xf64>
        %5 = affine.load %arg4[%arg6] : memref<40xf64>
        %6 = mulf %4, %5 : f64
        %7 = affine.load %1[0] : memref<1xf64>
        %8 = subf %7, %6 : f64
        affine.store %8, %1[0] : memref<1xf64>
      }
      %3 = affine.load %1[0] : memref<1xf64>
      affine.store %3, %arg4[%arg5] : memref<40xf64>
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg4[-%arg5 + symbol(%0) - 1] : memref<40xf64>
      affine.store %2, %1[0] : memref<1xf64>
      affine.for %arg6 = #map1(%arg5)[%0] to %0 {
        %6 = affine.load %arg1[-%arg5 + symbol(%0) - 1, %arg6] : memref<40x40xf64>
        %7 = affine.load %arg3[%arg6] : memref<40xf64>
        %8 = mulf %6, %7 : f64
        %9 = affine.load %1[0] : memref<1xf64>
        %10 = subf %9, %8 : f64
        affine.store %10, %1[0] : memref<1xf64>
      }
      %3 = affine.load %1[0] : memref<1xf64>
      %4 = affine.load %arg1[-%arg5 + symbol(%0) - 1, -%arg5 + symbol(%0) - 1] : memref<40x40xf64>
      %5 = divf %3, %4 : f64
      affine.store %5, %arg3[-%arg5 + symbol(%0) - 1] : memref<40xf64>
    }
    return
  }
  func private @print_array(%arg0: i32, %arg1: memref<40xf64>) {
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
    %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<2 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb2
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = remi_signed %13, %c20_i32 : i32
    %17 = cmpi "eq", %16, %c0_i32 : i32
    scf.if %17 {
      %38 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %39 = llvm.load %38 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %40 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
      %41 = llvm.getelementptr %40[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %42 = llvm.call @fprintf(%39, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %18 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %19 = llvm.load %18 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %20 = llvm.mlir.addressof @str4 : !llvm.ptr<array<8 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = load %arg1[%15] : memref<40xf64>
    %23 = llvm.mlir.cast %22 : f64 to !llvm.double
    %24 = llvm.call @fprintf(%19, %21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %25 = addi %13, %c1_i32 : i32
    br ^bb1(%25 : i32)
  ^bb3:  // pred: ^bb1
    %26 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %27 = llvm.load %26 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %28 = llvm.mlir.addressof @str5 : !llvm.ptr<array<17 x i8>>
    %29 = llvm.getelementptr %28[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %30 = llvm.mlir.addressof @str2 : !llvm.ptr<array<2 x i8>>
    %31 = llvm.getelementptr %30[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %32 = llvm.call @fprintf(%27, %29, %31) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %35 = llvm.mlir.addressof @str6 : !llvm.ptr<array<23 x i8>>
    %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %37 = llvm.call @fprintf(%34, %36) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  }
}

