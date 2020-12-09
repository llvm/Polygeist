module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("w\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c0 = constant 0 : index
    %c4000_i32 = constant 4000 : i32
    %c42_i32 = constant 42 : i32
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloc() : memref<4000x4000xf64>
    %3 = alloc() : memref<4000xf64>
    %4 = alloc() : memref<4000xf64>
    %5 = alloc() : memref<4000xf64>
    %6 = alloc() : memref<4000xf64>
    %7 = alloc() : memref<4000xf64>
    %8 = alloc() : memref<4000xf64>
    %9 = alloc() : memref<4000xf64>
    %10 = alloc() : memref<4000xf64>
    %11 = memref_cast %0 : memref<1xf64> to memref<?xf64>
    %12 = memref_cast %1 : memref<1xf64> to memref<?xf64>
    call @init_array(%c4000_i32, %11, %12, %2, %3, %4, %5, %6, %7, %8, %9, %10) : (i32, memref<?xf64>, memref<?xf64>, memref<4000x4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    %13 = load %0[%c0] : memref<1xf64>
    %14 = load %1[%c0] : memref<1xf64>
    call @kernel_gemver(%c4000_i32, %13, %14, %2, %3, %4, %5, %6, %7, %8, %9, %10) : (i32, f64, f64, memref<4000x4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>, memref<4000xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %15 = cmpi "sgt", %arg0, %c42_i32 : i32
    %16 = scf.if %15 -> (i1) {
      %17 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
      %18 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %19 = llvm.mlir.constant(0 : index) : !llvm.i64
      %20 = llvm.getelementptr %18[%19, %19] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %21 = llvm.call @strcmp(%17, %20) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
      %22 = llvm.mlir.cast %21 : !llvm.i32 to i32
      %23 = trunci %22 : i32 to i1
      %24 = xor %23, %true : i1
      scf.yield %24 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %16 {
      call @print_array(%c4000_i32, %7) : (i32, memref<4000xf64>) -> ()
    }
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<4000x4000xf64>, %arg4: memref<4000xf64>, %arg5: memref<4000xf64>, %arg6: memref<4000xf64>, %arg7: memref<4000xf64>, %arg8: memref<4000xf64>, %arg9: memref<4000xf64>, %arg10: memref<4000xf64>, %arg11: memref<4000xf64>) {
    %c0 = constant 0 : index
    %cst = constant 1.500000e+00 : f64
    %cst_0 = constant 1.200000e+00 : f64
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %cst_1 = constant 2.000000e+00 : f64
    %cst_2 = constant 4.000000e+00 : f64
    %cst_3 = constant 6.000000e+00 : f64
    %cst_4 = constant 8.000000e+00 : f64
    %cst_5 = constant 9.000000e+00 : f64
    %cst_6 = constant 0.000000e+00 : f64
    store %cst, %arg1[%c0] : memref<?xf64>
    store %cst_0, %arg2[%c0] : memref<?xf64>
    %0 = sitofp %arg0 : i32 to f64
    br ^bb1(%c0_i32 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb4
    %2 = cmpi "slt", %1, %arg0 : i32
    %3 = index_cast %1 : i32 to index
    cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %4 = sitofp %1 : i32 to f64
    store %4, %arg4[%3] : memref<4000xf64>
    %5 = addi %1, %c1_i32 : i32
    %6 = sitofp %5 : i32 to f64
    %7 = divf %6, %0 : f64
    %8 = divf %7, %cst_1 : f64
    store %8, %arg6[%3] : memref<4000xf64>
    %9 = divf %7, %cst_2 : f64
    store %9, %arg5[%3] : memref<4000xf64>
    %10 = divf %7, %cst_3 : f64
    store %10, %arg7[%3] : memref<4000xf64>
    %11 = divf %7, %cst_4 : f64
    store %11, %arg10[%3] : memref<4000xf64>
    %12 = divf %7, %cst_5 : f64
    store %12, %arg11[%3] : memref<4000xf64>
    store %cst_6, %arg9[%3] : memref<4000xf64>
    store %cst_6, %arg8[%3] : memref<4000xf64>
    br ^bb4(%c0_i32 : i32)
  ^bb3:  // pred: ^bb1
    return
  ^bb4(%13: i32):  // 2 preds: ^bb2, ^bb5
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb5, ^bb1(%5 : i32)
  ^bb5:  // pred: ^bb4
    %16 = muli %1, %13 : i32
    %17 = remi_signed %16, %arg0 : i32
    %18 = sitofp %17 : i32 to f64
    %19 = divf %18, %0 : f64
    store %19, %arg3[%3, %15] : memref<4000x4000xf64>
    %20 = addi %13, %c1_i32 : i32
    br ^bb4(%20 : i32)
  }
  func private @polybench_timer_start()
  func private @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<4000x4000xf64>, %arg4: memref<4000xf64>, %arg5: memref<4000xf64>, %arg6: memref<4000xf64>, %arg7: memref<4000xf64>, %arg8: memref<4000xf64>, %arg9: memref<4000xf64>, %arg10: memref<4000xf64>, %arg11: memref<4000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg3[%arg12, %arg13] : memref<4000x4000xf64>
        %2 = affine.load %arg4[%arg12] : memref<4000xf64>
        %3 = affine.load %arg5[%arg13] : memref<4000xf64>
        %4 = mulf %2, %3 : f64
        %5 = addf %1, %4 : f64
        %6 = affine.load %arg6[%arg12] : memref<4000xf64>
        %7 = affine.load %arg7[%arg13] : memref<4000xf64>
        %8 = mulf %6, %7 : f64
        %9 = addf %5, %8 : f64
        affine.store %9, %arg3[%arg12, %arg13] : memref<4000x4000xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg9[%arg12] : memref<4000xf64>
        %2 = affine.load %arg3[%arg13, %arg12] : memref<4000x4000xf64>
        %3 = mulf %arg2, %2 : f64
        %4 = affine.load %arg10[%arg13] : memref<4000xf64>
        %5 = mulf %3, %4 : f64
        %6 = addf %1, %5 : f64
        affine.store %6, %arg9[%arg12] : memref<4000xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<4000xf64>
      %2 = affine.load %arg11[%arg12] : memref<4000xf64>
      %3 = addf %1, %2 : f64
      affine.store %3, %arg9[%arg12] : memref<4000xf64>
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg8[%arg12] : memref<4000xf64>
        %2 = affine.load %arg3[%arg12, %arg13] : memref<4000x4000xf64>
        %3 = mulf %arg1, %2 : f64
        %4 = affine.load %arg9[%arg13] : memref<4000xf64>
        %5 = mulf %3, %4 : f64
        %6 = addf %1, %5 : f64
        affine.store %6, %arg8[%arg12] : memref<4000xf64>
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @print_array(%arg0: i32, %arg1: memref<4000xf64>) {
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.mlir.constant(0 : index) : !llvm.i64
    %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
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
      %40 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
      %41 = llvm.getelementptr %40[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %42 = llvm.call @fprintf(%39, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %18 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %19 = llvm.load %18 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %20 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = load %arg1[%15] : memref<4000xf64>
    %23 = llvm.mlir.cast %22 : f64 to !llvm.double
    %24 = llvm.call @fprintf(%19, %21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %25 = addi %13, %c1_i32 : i32
    br ^bb1(%25 : i32)
  ^bb3:  // pred: ^bb1
    %26 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %27 = llvm.load %26 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %28 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %29 = llvm.getelementptr %28[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %30 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %31 = llvm.getelementptr %30[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %32 = llvm.call @fprintf(%27, %29, %31) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %35 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %37 = llvm.call @fprintf(%34, %36) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  }
}

