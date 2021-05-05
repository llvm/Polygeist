module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("D\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1600_i32 = constant 1600 : i32
    %c1800_i32 = constant 1800 : i32
    %c2200_i32 = constant 2200 : i32
    %c2400_i32 = constant 2400 : i32
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = memref.alloca() : memref<1xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = memref.alloc() : memref<1600x1800xf64>
    %3 = memref.alloc() : memref<1600x2200xf64>
    %4 = memref.alloc() : memref<2200x1800xf64>
    %5 = memref.alloc() : memref<1800x2400xf64>
    %6 = memref.alloc() : memref<1600x2400xf64>
    %7 = memref.cast %0 : memref<1xf64> to memref<?xf64>
    %8 = memref.cast %1 : memref<1xf64> to memref<?xf64>
    %9 = memref.cast %3 : memref<1600x2200xf64> to memref<?x2200xf64>
    %10 = memref.cast %4 : memref<2200x1800xf64> to memref<?x1800xf64>
    %11 = memref.cast %5 : memref<1800x2400xf64> to memref<?x2400xf64>
    %12 = memref.cast %6 : memref<1600x2400xf64> to memref<?x2400xf64>
    call @init_array(%c1600_i32, %c1800_i32, %c2200_i32, %c2400_i32, %7, %8, %9, %10, %11, %12) : (i32, i32, i32, i32, memref<?xf64>, memref<?xf64>, memref<?x2200xf64>, memref<?x1800xf64>, memref<?x2400xf64>, memref<?x2400xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    %13 = affine.load %0[0] : memref<1xf64>
    %14 = affine.load %1[0] : memref<1xf64>
    %15 = memref.cast %2 : memref<1600x1800xf64> to memref<?x1800xf64>
    call @kernel_2mm(%c1600_i32, %c1800_i32, %c2200_i32, %c2400_i32, %13, %14, %15, %9, %10, %11, %12) : (i32, i32, i32, i32, f64, f64, memref<?x1800xf64>, memref<?x2200xf64>, memref<?x1800xf64>, memref<?x2400xf64>, memref<?x2400xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %16 = cmpi sgt, %arg0, %c42_i32 : i32
    %17 = scf.if %16 -> (i1) {
      %18 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %19 = llvm.load %18 : !llvm.ptr<ptr<i8>>
      %20 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %21 = llvm.getelementptr %20[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %22 = llvm.call @strcmp(%19, %21) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %23 = trunci %22 : i32 to i1
      %24 = xor %23, %true : i1
      scf.yield %24 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %17 {
      call @print_array(%c1600_i32, %c2400_i32, %12) : (i32, i32, memref<?x2400xf64>) -> ()
    }
    memref.dealloc %2 : memref<1600x1800xf64>
    memref.dealloc %3 : memref<1600x2200xf64>
    memref.dealloc %4 : memref<2200x1800xf64>
    memref.dealloc %5 : memref<1800x2400xf64>
    memref.dealloc %6 : memref<1600x2400xf64>
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?x2200xf64>, %arg7: memref<?x1800xf64>, %arg8: memref<?x2400xf64>, %arg9: memref<?x2400xf64>) {
    %cst = constant 1.500000e+00 : f64
    %cst_0 = constant 1.200000e+00 : f64
    %c3_i32 = constant 3 : i32
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    affine.store %cst, %arg4[0] : memref<?xf64>
    affine.store %cst_0, %arg5[0] : memref<?xf64>
    %0:2 = scf.while (%arg10 = %c0_i32) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg10, %arg0 : i32
      scf.condition(%4) %c0_i32, %arg10 : i32, i32
    } do {
    ^bb0(%arg10: i32, %arg11: i32):  // no predecessors
      %4 = index_cast %arg11 : i32 to index
      %5 = scf.while (%arg12 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg12, %arg2 : i32
        scf.condition(%7) %arg12 : i32
      } do {
      ^bb0(%arg12: i32):  // no predecessors
        %7 = index_cast %arg12 : i32 to index
        %8 = muli %arg11, %arg12 : i32
        %9 = addi %8, %c1_i32 : i32
        %10 = remi_signed %9, %arg0 : i32
        %11 = sitofp %10 : i32 to f64
        %12 = sitofp %arg0 : i32 to f64
        %13 = divf %11, %12 : f64
        memref.store %13, %arg6[%4, %7] : memref<?x2200xf64>
        %14 = addi %arg12, %c1_i32 : i32
        scf.yield %14 : i32
      }
      %6 = addi %arg11, %c1_i32 : i32
      scf.yield %6 : i32
    }
    %1:2 = scf.while (%arg10 = %0#0) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg10, %arg2 : i32
      scf.condition(%4) %c0_i32, %arg10 : i32, i32
    } do {
    ^bb0(%arg10: i32, %arg11: i32):  // no predecessors
      %4 = index_cast %arg11 : i32 to index
      %5 = index_cast %arg1 : i32 to index
      %6 = scf.for %arg12 = %c0 to %5 step %c1 iter_args(%arg13 = %c0_i32) -> (i32) {
        %8 = index_cast %arg13 : i32 to index
        %9 = addi %arg13, %c1_i32 : i32
        %10 = muli %arg11, %9 : i32
        %11 = remi_signed %10, %arg1 : i32
        %12 = sitofp %11 : i32 to f64
        %13 = sitofp %arg1 : i32 to f64
        %14 = divf %12, %13 : f64
        memref.store %14, %arg7[%4, %8] : memref<?x1800xf64>
        scf.yield %9 : i32
      }
      %7 = addi %arg11, %c1_i32 : i32
      scf.yield %7 : i32
    }
    %2:2 = scf.while (%arg10 = %1#0) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg10, %arg1 : i32
      scf.condition(%4) %c0_i32, %arg10 : i32, i32
    } do {
    ^bb0(%arg10: i32, %arg11: i32):  // no predecessors
      %4 = index_cast %arg11 : i32 to index
      %5 = scf.while (%arg12 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg12, %arg3 : i32
        scf.condition(%7) %arg12 : i32
      } do {
      ^bb0(%arg12: i32):  // no predecessors
        %7 = index_cast %arg12 : i32 to index
        %8 = addi %arg12, %c3_i32 : i32
        %9 = muli %arg11, %8 : i32
        %10 = addi %9, %c1_i32 : i32
        %11 = remi_signed %10, %arg3 : i32
        %12 = sitofp %11 : i32 to f64
        %13 = sitofp %arg3 : i32 to f64
        %14 = divf %12, %13 : f64
        memref.store %14, %arg8[%4, %7] : memref<?x2400xf64>
        %15 = addi %arg12, %c1_i32 : i32
        scf.yield %15 : i32
      }
      %6 = addi %arg11, %c1_i32 : i32
      scf.yield %6 : i32
    }
    %3 = scf.while (%arg10 = %2#0) : (i32) -> i32 {
      %4 = cmpi slt, %arg10, %arg0 : i32
      scf.condition(%4) %arg10 : i32
    } do {
    ^bb0(%arg10: i32):  // no predecessors
      %4 = index_cast %arg10 : i32 to index
      %5 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg11, %arg3 : i32
        scf.condition(%7) %arg11 : i32
      } do {
      ^bb0(%arg11: i32):  // no predecessors
        %7 = index_cast %arg11 : i32 to index
        %8 = addi %arg11, %c2_i32 : i32
        %9 = muli %arg10, %8 : i32
        %10 = remi_signed %9, %arg2 : i32
        %11 = sitofp %10 : i32 to f64
        %12 = sitofp %arg2 : i32 to f64
        %13 = divf %11, %12 : f64
        memref.store %13, %arg9[%4, %7] : memref<?x2400xf64>
        %14 = addi %arg11, %c1_i32 : i32
        scf.yield %14 : i32
      }
      %6 = addi %arg10, %c1_i32 : i32
      scf.yield %6 : i32
    }
    return
  }
  func private @polybench_timer_start()
  func private @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x1800xf64>, %arg7: memref<?x2200xf64>, %arg8: memref<?x1800xf64>, %arg9: memref<?x2400xf64>, %arg10: memref<?x2400xf64>) {
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg3 : i32 to index
    %3 = index_cast %arg0 : i32 to index
    affine.for %arg11 = 0 to %3 {
      affine.for %arg12 = 0 to %0 {
        affine.store %cst, %arg6[%arg11, %arg12] : memref<?x1800xf64>
        affine.for %arg13 = 0 to %1 {
          %4 = affine.load %arg7[%arg11, %arg13] : memref<?x2200xf64>
          %5 = mulf %arg4, %4 {scop.splittable = 1 : index} : f64
          %6 = affine.load %arg8[%arg13, %arg12] : memref<?x1800xf64>
          %7 = mulf %5, %6 {scop.splittable = 0 : index} : f64
          %8 = affine.load %arg6[%arg11, %arg12] : memref<?x1800xf64>
          %9 = addf %8, %7 : f64
          affine.store %9, %arg6[%arg11, %arg12] : memref<?x1800xf64>
        }
      }
    }
    affine.for %arg11 = 0 to %3 {
      affine.for %arg12 = 0 to %2 {
        %4 = affine.load %arg10[%arg11, %arg12] : memref<?x2400xf64>
        %5 = mulf %4, %arg5 : f64
        affine.store %5, %arg10[%arg11, %arg12] : memref<?x2400xf64>
        affine.for %arg13 = 0 to %0 {
          %6 = affine.load %arg6[%arg11, %arg13] : memref<?x1800xf64>
          %7 = affine.load %arg9[%arg13, %arg12] : memref<?x2400xf64>
          %8 = mulf %6, %7 {scop.splittable = 2 : index} : f64
          %9 = affine.load %arg10[%arg11, %arg12] : memref<?x2400xf64>
          %10 = addf %9, %8 : f64
          affine.store %10, %arg10[%arg11, %arg12] : memref<?x2400xf64>
        }
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<?x2400xf64>) {
    %c0_i64 = constant 0 : i64
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.getelementptr %2[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %4 = llvm.call @fprintf(%1, %3) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    %5 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %6 = llvm.load %5 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %8 = llvm.getelementptr %7[%c0_i64, %c0_i64] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %9 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %10 = llvm.getelementptr %9[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %11 = llvm.call @fprintf(%6, %8, %10) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %12 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
      %25 = cmpi slt, %arg3, %arg0 : i32
      scf.condition(%25) %arg3 : i32
    } do {
    ^bb0(%arg3: i32):  // no predecessors
      %25 = index_cast %arg3 : i32 to index
      %26 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %28 = cmpi slt, %arg4, %arg1 : i32
        scf.condition(%28) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %28 = index_cast %arg4 : i32 to index
        %29 = muli %arg3, %arg0 : i32
        %30 = addi %29, %arg4 : i32
        %31 = remi_signed %30, %c20_i32 : i32
        %32 = cmpi eq, %31, %c0_i32 : i32
        scf.if %32 {
          %40 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %41 = llvm.load %40 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %42 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
          %43 = llvm.getelementptr %42[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %44 = llvm.call @fprintf(%41, %43) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
        }
        %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %35 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
        %36 = llvm.getelementptr %35[%c0_i64, %c0_i64] : (!llvm.ptr<array<8 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %37 = memref.load %arg2[%25, %28] : memref<?x2400xf64>
        %38 = llvm.call @fprintf(%34, %36, %37) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %39 = addi %arg4, %c1_i32 : i32
        scf.yield %39 : i32
      }
      %27 = addi %arg3, %c1_i32 : i32
      scf.yield %27 : i32
    }
    %13 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %14 = llvm.load %13 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %15 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %16 = llvm.getelementptr %15[%c0_i64, %c0_i64] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %18 = llvm.getelementptr %17[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %19 = llvm.call @fprintf(%14, %16, %18) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %20 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %22 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %23 = llvm.getelementptr %22[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %24 = llvm.call @fprintf(%21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    return
  }
}

