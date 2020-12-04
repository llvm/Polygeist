#map = affine_map<(d0) -> (d0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2lf \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("C\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1000_i32 = constant 1000 : i32
    %c1200_i32 = constant 1200 : i32
    %cst = constant 1.500000e+00 : f64
    %cst_0 = constant 1.200000e+00 : f64
    %c0_i32 = constant 0 : i32
    %c100_i32 = constant 100 : i32
    %c1_i32 = constant 1 : i32
    %c-999_i32 = constant -999 : i32
    %c0 = constant 0 : index
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloc() : memref<1000x1200xf64>
    %3 = alloc() : memref<1000x1000xf64>
    %4 = alloc() : memref<1000x1200xf64>
    store %cst, %0[%c0] : memref<1xf64>
    store %cst_0, %1[%c0] : memref<1xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb4
    %6 = cmpi "slt", %5, %c1000_i32 : i32
    %7 = index_cast %5 : i32 to index
    cond_br %6, ^bb2(%c0_i32 : i32), ^bb5(%c0_i32 : i32)
  ^bb2(%8: i32):  // 2 preds: ^bb1, ^bb3
    %9 = cmpi "slt", %8, %c1200_i32 : i32
    %10 = index_cast %8 : i32 to index
    cond_br %9, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %11 = addi %5, %8 : i32
    %12 = remi_signed %11, %c100_i32 : i32
    %13 = sitofp %12 : i32 to f64
    %14 = sitofp %c1000_i32 : i32 to f64
    %15 = divf %13, %14 : f64
    store %15, %2[%7, %10] : memref<1000x1200xf64>
    %16 = addi %5, %c1200_i32 : i32
    %17 = subi %16, %8 : i32
    %18 = remi_signed %17, %c100_i32 : i32
    %19 = sitofp %18 : i32 to f64
    %20 = divf %19, %14 : f64
    store %20, %4[%7, %10] : memref<1000x1200xf64>
    %21 = addi %8, %c1_i32 : i32
    br ^bb2(%21 : i32)
  ^bb4:  // pred: ^bb2
    %22 = addi %5, %c1_i32 : i32
    br ^bb1(%22 : i32)
  ^bb5(%23: i32):  // 2 preds: ^bb1, ^bb10
    %24 = cmpi "slt", %23, %c1000_i32 : i32
    %25 = index_cast %23 : i32 to index
    cond_br %24, ^bb7(%c0_i32 : i32), ^bb6
  ^bb6:  // pred: ^bb5
    %26 = load %0[%c0] : memref<1xf64>
    %27 = load %1[%c0] : memref<1xf64>
    %28 = alloca() : memref<1xf64>
    affine.for %arg2 = 0 to 1200 {
      call @S0(%28) : (memref<1xf64>) -> ()
      call @S3(%2, %c0, %arg2, %26, %28, %3, %4, %27) : (memref<1000x1200xf64>, index, index, f64, memref<1xf64>, memref<1000x1000xf64>, memref<1000x1200xf64>, f64) -> ()
    }
    affine.for %arg2 = 1 to 1000 {
      affine.for %arg3 = 0 to 1200 {
        call @S1(%2, %arg2, %arg3, %3, %c0, %26, %4) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, f64, memref<1000x1200xf64>) -> ()
        call @S0(%28) : (memref<1xf64>) -> ()
        call @S2(%28, %3, %arg2, %arg3, %4, %c0) : (memref<1xf64>, memref<1000x1000xf64>, index, index, memref<1000x1200xf64>, index) -> ()
        affine.for %arg4 = 1 to #map(%arg2) {
          call @S1(%2, %arg2, %arg3, %3, %arg4, %26, %4) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, f64, memref<1000x1200xf64>) -> ()
          call @S2(%28, %3, %arg2, %arg3, %4, %arg4) : (memref<1xf64>, memref<1000x1000xf64>, index, index, memref<1000x1200xf64>, index) -> ()
        }
        call @S3(%2, %arg2, %arg3, %26, %28, %3, %4, %27) : (memref<1000x1200xf64>, index, index, f64, memref<1xf64>, memref<1000x1000xf64>, memref<1000x1200xf64>, f64) -> ()
      }
    }
    call @print_array(%c1000_i32, %c1200_i32, %2) : (i32, i32, memref<1000x1200xf64>) -> ()
    return %c0_i32 : i32
  ^bb7(%29: i32):  // 2 preds: ^bb5, ^bb8
    %30 = cmpi "sle", %29, %23 : i32
    %31 = index_cast %29 : i32 to index
    cond_br %30, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %32 = addi %23, %29 : i32
    %33 = remi_signed %32, %c100_i32 : i32
    %34 = sitofp %33 : i32 to f64
    %35 = sitofp %c1000_i32 : i32 to f64
    %36 = divf %34, %35 : f64
    store %36, %3[%25, %31] : memref<1000x1000xf64>
    %37 = addi %29, %c1_i32 : i32
    br ^bb7(%37 : i32)
  ^bb9:  // pred: ^bb7
    %38 = addi %23, %c1_i32 : i32
    br ^bb10(%38 : i32)
  ^bb10(%39: i32):  // 2 preds: ^bb9, ^bb11
    %40 = cmpi "slt", %39, %c1000_i32 : i32
    %41 = index_cast %39 : i32 to index
    cond_br %40, ^bb11, ^bb5(%38 : i32)
  ^bb11:  // pred: ^bb10
    %42 = sitofp %c-999_i32 : i32 to f64
    store %42, %3[%25, %41] : memref<1000x1000xf64>
    %43 = addi %39, %c1_i32 : i32
    br ^bb10(%43 : i32)
  }
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>) {
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
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %16 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %17 = llvm.load %16 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %18 = llvm.mlir.addressof @str5 : !llvm.ptr<array<17 x i8>>
    %19 = llvm.getelementptr %18[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %20 = llvm.mlir.addressof @str2 : !llvm.ptr<array<2 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
    %31 = muli %13, %arg0 : i32
    %32 = addi %31, %28 : i32
    %33 = remi_signed %32, %c20_i32 : i32
    %34 = cmpi "eq", %33, %c0_i32 : i32
    scf.if %34 {
      %44 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %45 = llvm.load %44 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %46 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
      %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str4 : !llvm.ptr<array<8 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg2[%15, %30] : memref<1000x1200xf64>
    %40 = llvm.mlir.cast %39 : f64 to !llvm.double
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %42 = addi %28, %c1_i32 : i32
    br ^bb3(%42 : i32)
  ^bb5:  // pred: ^bb3
    %43 = addi %13, %c1_i32 : i32
    br ^bb1(%43 : i32)
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %c0_i32 = constant 0 : i32
    %0 = sitofp %c0_i32 : i32 to f64
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func private @S1(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1000xf64>, %arg4: index, %arg5: f64, %arg6: memref<1000x1200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %1 = affine.load %arg6[%arg4, %arg2] : memref<1000x1200xf64>
    %2 = mulf %arg5, %1 : f64
    %3 = affine.load %arg3[%arg4, %arg1] : memref<1000x1000xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
  func private @S2(%arg0: memref<1xf64>, %arg1: memref<1000x1000xf64>, %arg2: index, %arg3: index, %arg4: memref<1000x1200xf64>, %arg5: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg4[%arg3, %arg5] : memref<1000x1200xf64>
    %2 = affine.load %arg1[%arg2, %arg3] : memref<1000x1000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func private @S3(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: f64, %arg4: memref<1xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>, %arg7: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    %1 = mulf %arg7, %0 : f64
    %2 = affine.load %arg6[%arg1, %arg2] : memref<1000x1200xf64>
    %3 = mulf %arg3, %2 : f64
    %4 = affine.load %arg5[%arg1, %arg1] : memref<1000x1000xf64>
    %5 = mulf %3, %4 : f64
    %6 = addf %1, %5 : f64
    %7 = affine.load %arg4[0] : memref<1xf64>
    %8 = mulf %arg3, %7 : f64
    %9 = addf %6, %8 : f64
    affine.store %9, %arg0[%arg1, %arg2] : memref<1000x1200xf64>
    return
  }
  func @kernel_symm_new(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
    %c0 = constant 0 : index
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %1 {
      call @S0(%0) : (memref<1xf64>) -> ()
      call @S3(%arg4, %c0, %arg7, %arg2, %0, %arg5, %arg6, %arg3) : (memref<1000x1200xf64>, index, index, f64, memref<1xf64>, memref<1000x1000xf64>, memref<1000x1200xf64>, f64) -> ()
    }
    affine.for %arg7 = 1 to %2 {
      affine.for %arg8 = 0 to %1 {
        call @S1(%arg4, %arg7, %arg8, %arg5, %c0, %arg2, %arg6) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, f64, memref<1000x1200xf64>) -> ()
        call @S0(%0) : (memref<1xf64>) -> ()
        call @S2(%0, %arg5, %arg7, %arg8, %arg6, %c0) : (memref<1xf64>, memref<1000x1000xf64>, index, index, memref<1000x1200xf64>, index) -> ()
        affine.for %arg9 = 1 to #map(%arg7) {
          call @S1(%arg4, %arg7, %arg8, %arg5, %arg9, %arg2, %arg6) : (memref<1000x1200xf64>, index, index, memref<1000x1000xf64>, index, f64, memref<1000x1200xf64>) -> ()
          call @S2(%0, %arg5, %arg7, %arg8, %arg6, %arg9) : (memref<1xf64>, memref<1000x1000xf64>, index, index, memref<1000x1200xf64>, index) -> ()
        }
        call @S3(%arg4, %arg7, %arg8, %arg2, %0, %arg5, %arg6, %arg3) : (memref<1000x1200xf64>, index, index, f64, memref<1xf64>, memref<1000x1000xf64>, memref<1000x1200xf64>, f64) -> ()
      }
    }
    return
  }
}

