#map = affine_map<(d0) -> (d0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("A\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c4000_i32 = constant 4000 : i32
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<4000x4000xf64>
    %1 = memref.cast %0 : memref<4000x4000xf64> to memref<?x4000xf64>
    call @init_array(%c4000_i32, %1) : (i32, memref<?x4000xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    call @kernel_cholesky(%c4000_i32, %1) : (i32, memref<?x4000xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %2 = cmpi sgt, %arg0, %c42_i32 : i32
    %3 = scf.if %2 -> (i1) {
      %4 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %5 = llvm.load %4 : !llvm.ptr<ptr<i8>>
      %6 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %7 = llvm.getelementptr %6[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %8 = llvm.call @strcmp(%5, %7) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %9 = trunci %8 : i32 to i1
      %10 = xor %9, %true : i1
      scf.yield %10 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %3 {
      call @print_array(%c4000_i32, %1) : (i32, memref<?x4000xf64>) -> ()
    }
    memref.dealloc %0 : memref<4000x4000xf64>
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: memref<?x4000xf64>) {
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %3 = cmpi slt, %arg2, %arg0 : i32
      scf.condition(%3) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %3 = index_cast %arg2 : i32 to index
      %4 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %10 = cmpi sle, %arg3, %arg2 : i32
        scf.condition(%10) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %10 = index_cast %arg3 : i32 to index
        %11 = subi %c0_i32, %arg3 : i32
        %12 = remi_signed %11, %arg0 : i32
        %13 = sitofp %12 : i32 to f64
        %14 = sitofp %arg0 : i32 to f64
        %15 = divf %13, %14 : f64
        %16 = sitofp %c1_i32 : i32 to f64
        %17 = addf %15, %16 : f64
        memref.store %17, %arg1[%3, %10] : memref<?x4000xf64>
        %18 = addi %arg3, %c1_i32 : i32
        scf.yield %18 : i32
      }
      %5 = addi %arg2, %c1_i32 : i32
      %6 = index_cast %arg0 : i32 to index
      %7 = addi %3, %c1 : index
      %8 = scf.for %arg3 = %7 to %6 step %c1 iter_args(%arg4 = %5) -> (i32) {
        %10 = index_cast %arg4 : i32 to index
        %11 = sitofp %c0_i32 : i32 to f64
        memref.store %11, %arg1[%3, %10] : memref<?x4000xf64>
        %12 = addi %arg4, %c1_i32 : i32
        scf.yield %12 : i32
      }
      %9 = sitofp %c1_i32 : i32 to f64
      memref.store %9, %arg1[%3, %3] : memref<?x4000xf64>
      scf.yield %5 : i32
    }
    %1 = memref.alloc() : memref<4000x4000xf64>
    %2 = index_cast %arg0 : i32 to index
    scf.for %arg2 = %c0 to %2 step %c1 {
      scf.for %arg3 = %c0 to %2 step %c1 {
        %3 = sitofp %c0_i32 : i32 to f64
        memref.store %3, %1[%arg2, %arg3] : memref<4000x4000xf64>
      }
    }
    scf.for %arg2 = %c0 to %2 step %c1 {
      scf.for %arg3 = %c0 to %2 step %c1 {
        scf.for %arg4 = %c0 to %2 step %c1 {
          %3 = memref.load %arg1[%arg3, %arg2] : memref<?x4000xf64>
          %4 = memref.load %arg1[%arg4, %arg2] : memref<?x4000xf64>
          %5 = mulf %3, %4 : f64
          %6 = memref.load %1[%arg3, %arg4] : memref<4000x4000xf64>
          %7 = addf %6, %5 : f64
          memref.store %7, %1[%arg3, %arg4] : memref<4000x4000xf64>
        }
      }
    }
    scf.for %arg2 = %c0 to %2 step %c1 {
      scf.for %arg3 = %c0 to %2 step %c1 {
        %3 = memref.load %1[%arg2, %arg3] : memref<4000x4000xf64>
        memref.store %3, %arg1[%arg2, %arg3] : memref<?x4000xf64>
      }
    }
    memref.dealloc %1 : memref<4000x4000xf64>
    return
  }
  func private @polybench_timer_start()
  func private @kernel_cholesky(%arg0: i32, %arg1: memref<?x4000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        affine.for %arg4 = 0 to #map(%arg3) {
          %6 = affine.load %arg1[%arg2, %arg4] : memref<?x4000xf64>
          %7 = affine.load %arg1[%arg3, %arg4] : memref<?x4000xf64>
          %8 = mulf %6, %7 : f64
          %9 = affine.load %arg1[%arg2, %arg3] : memref<?x4000xf64>
          %10 = subf %9, %8 : f64
          affine.store %10, %arg1[%arg2, %arg3] : memref<?x4000xf64>
        }
        %3 = affine.load %arg1[%arg3, %arg3] : memref<?x4000xf64>
        %4 = affine.load %arg1[%arg2, %arg3] : memref<?x4000xf64>
        %5 = divf %4, %3 : f64
        affine.store %5, %arg1[%arg2, %arg3] : memref<?x4000xf64>
      }
      affine.for %arg3 = 0 to #map(%arg2) {
        %3 = affine.load %arg1[%arg2, %arg3] : memref<?x4000xf64>
        %4 = mulf %3, %3 : f64
        %5 = affine.load %arg1[%arg2, %arg2] : memref<?x4000xf64>
        %6 = subf %5, %4 : f64
        affine.store %6, %arg1[%arg2, %arg2] : memref<?x4000xf64>
      }
      %1 = affine.load %arg1[%arg2, %arg2] : memref<?x4000xf64>
      %2 = math.sqrt %1 : f64
      affine.store %2, %arg1[%arg2, %arg2] : memref<?x4000xf64>
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @print_array(%arg0: i32, %arg1: memref<?x4000xf64>) {
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
    %12 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %25 = cmpi slt, %arg2, %arg0 : i32
      scf.condition(%25) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %25 = index_cast %arg2 : i32 to index
      %26 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %28 = cmpi sle, %arg3, %arg2 : i32
        scf.condition(%28) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %28 = index_cast %arg3 : i32 to index
        %29 = muli %arg2, %arg0 : i32
        %30 = addi %29, %arg3 : i32
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
        %37 = memref.load %arg1[%25, %28] : memref<?x4000xf64>
        %38 = llvm.call @fprintf(%34, %36, %37) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %39 = addi %arg3, %c1_i32 : i32
        scf.yield %39 : i32
      }
      %27 = addi %arg2, %c1_i32 : i32
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
