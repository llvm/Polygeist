#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0, d1) -> (-d0 + d1)>
#map4 = affine_map<(d0) -> (d0 * 3)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str9("%0.6f\0A\00")
  global_memref "private" @polybench_t_end : memref<1xf64>
  llvm.mlir.global internal constant @str8("Error return from gettimeofday: %d\00")
  llvm.func @printf(!llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.func @gettimeofday(!llvm.ptr<struct<"struct.timeval", (i64, i64)>>, !llvm.ptr<struct<"struct.timezone", (i32, i32)>>) -> !llvm.i32
  global_memref "private" @polybench_t_start : memref<1xf64>
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("y\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c2000_i32 = constant 2000 : i32
    %c42_i32 = constant 42 : i32
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<2000xf64>
    %1 = alloc() : memref<2000xf64>
    call @init_array(%c2000_i32, %0) : (i32, memref<2000xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    call @kernel_durbin(%c2000_i32, %0, %1) : (i32, memref<2000xf64>, memref<2000xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %2 = cmpi "sgt", %arg0, %c42_i32 : i32
    %3 = scf.if %2 -> (i1) {
      %4 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
      %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %6 = llvm.mlir.constant(0 : index) : !llvm.i64
      %7 = llvm.getelementptr %5[%6, %6] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %8 = llvm.call @strcmp(%4, %7) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
      %9 = llvm.mlir.cast %8 : !llvm.i32 to i32
      %10 = trunci %9 : i32 to i1
      %11 = xor %10, %true : i1
      scf.yield %11 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %3 {
      call @print_array(%c2000_i32, %1) : (i32, memref<2000xf64>) -> ()
    }
    return %c0_i32 : i32
  }
  func @init_array(%arg0: i32, %arg1: memref<2000xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = cmpi "slt", %0, %arg0 : i32
    %2 = index_cast %0 : i32 to index
    cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %3 = addi %arg0, %c1_i32 : i32
    %4 = subi %3, %0 : i32
    %5 = sitofp %4 : i32 to f64
    store %5, %arg1[%2] : memref<2000xf64>
    %6 = addi %0, %c1_i32 : i32
    br ^bb1(%6 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func @polybench_timer_start() {
    %c0 = constant 0 : index
    call @polybench_prepare_instruments() : () -> ()
    %0 = get_global_memref @polybench_t_start : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @kernel_durbin(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
    %c0 = constant 0 : index
    %cst = constant 1.000000e+00 : f64
    %0 = alloca() : memref<2000xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = load %arg1[%c0] : memref<2000xf64>
    %5 = negf %4 : f64
    store %5, %arg2[%c0] : memref<2000xf64>
    store %cst, %2[%c0] : memref<1xf64>
    %6 = load %arg1[%c0] : memref<2000xf64>
    %7 = negf %6 : f64
    store %7, %1[%c0] : memref<1xf64>
    %8 = index_cast %arg0 : i32 to index
    affine.for %arg3 = 1 to %8 {
      call @S0(%2, %1) : (memref<1xf64>, memref<1xf64>) -> ()
      call @S1(%3) : (memref<1xf64>) -> ()
      affine.for %arg4 = 0 to #map0(%arg3) {
        call @S2(%3, %arg2, %arg4, %arg1, %arg3) : (memref<1xf64>, memref<2000xf64>, index, memref<2000xf64>, index) -> ()
      }
      %9 = alloca() : memref<1xf64>
      call @S3(%9, %2, %1, %3, %arg1, %arg3) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S4(%1, %2, %3, %arg1, %arg3) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg4 = 0 to #map0(%arg3) {
        call @S5(%arg4, %0, %arg2, %arg3, %9) : (index, memref<2000xf64>, memref<2000xf64>, index, memref<1xf64>) -> ()
      }
      affine.for %arg4 = 0 to #map0(%arg3) {
        call @S6(%arg2, %arg4, %0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
      call @S7(%arg2, %arg3, %2, %1, %3, %arg1) : (memref<2000xf64>, index, memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>) -> ()
    }
    return
  }
  func @polybench_timer_stop() {
    %c0 = constant 0 : index
    %0 = get_global_memref @polybench_t_end : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @polybench_timer_print() {
    %c0 = constant 0 : index
    %0 = llvm.mlir.addressof @str9 : !llvm.ptr<array<7 x i8>>
    %1 = llvm.mlir.constant(0 : index) : !llvm.i64
    %2 = llvm.getelementptr %0[%1, %1] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %3 = get_global_memref @polybench_t_end : memref<1xf64>
    %4 = load %3[%c0] : memref<1xf64>
    %5 = get_global_memref @polybench_t_start : memref<1xf64>
    %6 = load %5[%c0] : memref<1xf64>
    %7 = subf %4, %6 : f64
    %8 = llvm.mlir.cast %7 : f64 to !llvm.double
    %9 = llvm.call @printf(%2, %8) : (!llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    return
  }
  func @print_array(%arg0: i32, %arg1: memref<2000xf64>) {
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.mlir.constant(0 : index) : !llvm.i64
    %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb2
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = remi_signed %13, %c20_i32 : i32
    %17 = cmpi "eq", %16, %c0_i32 : i32
    scf.if %17 {
      %38 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %39 = llvm.load %38 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %40 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
      %41 = llvm.getelementptr %40[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %42 = llvm.call @fprintf(%39, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %18 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %19 = llvm.load %18 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %20 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = load %arg1[%15] : memref<2000xf64>
    %23 = llvm.mlir.cast %22 : f64 to !llvm.double
    %24 = llvm.call @fprintf(%19, %21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %25 = addi %13, %c1_i32 : i32
    br ^bb1(%25 : i32)
  ^bb3:  // pred: ^bb1
    %26 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %27 = llvm.load %26 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %28 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %29 = llvm.getelementptr %28[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %30 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %31 = llvm.getelementptr %30[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %32 = llvm.call @fprintf(%27, %29, %31) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %35 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %37 = llvm.call @fprintf(%34, %36) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  }
  func private @free(memref<?xi8>)
  func @polybench_prepare_instruments() {
    return
  }
  func @rtclock() -> f64 {
    %c0_i32 = constant 0 : i32
    %cst = constant 9.9999999999999995E-7 : f64
    %0 = llvm.mlir.constant(1 : index) : !llvm.i64
    %1 = llvm.alloca %0 x !llvm.struct<"struct.timeval", (i64, i64)> : (!llvm.i64) -> !llvm.ptr<struct<"struct.timeval", (i64, i64)>>
    %2 = llvm.mlir.null : !llvm.ptr<struct<"struct.timezone", (i32, i32)>>
    %3 = llvm.call @gettimeofday(%1, %2) : (!llvm.ptr<struct<"struct.timeval", (i64, i64)>>, !llvm.ptr<struct<"struct.timezone", (i32, i32)>>) -> !llvm.i32
    %4 = llvm.mlir.cast %3 : !llvm.i32 to i32
    %5 = llvm.load %1 : !llvm.ptr<struct<"struct.timeval", (i64, i64)>>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<"struct.timeval", (i64, i64)>
    %7 = llvm.mlir.cast %6 : !llvm.i64 to i64
    %8 = llvm.extractvalue %5[1] : !llvm.struct<"struct.timeval", (i64, i64)>
    %9 = llvm.mlir.cast %8 : !llvm.i64 to i64
    %10 = cmpi "ne", %4, %c0_i32 : i32
    scf.if %10 {
      %15 = llvm.mlir.addressof @str8 : !llvm.ptr<array<35 x i8>>
      %16 = llvm.mlir.constant(0 : index) : !llvm.i64
      %17 = llvm.getelementptr %15[%16, %16] : (!llvm.ptr<array<35 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %18 = llvm.mlir.cast %4 : i32 to !llvm.i32
      %19 = llvm.call @printf(%17, %18) : (!llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
    }
    %11 = sitofp %7 : i64 to f64
    %12 = sitofp %9 : i64 to f64
    %13 = mulf %12, %cst : f64
    %14 = addf %11, %13 : f64
    return %14 : f64
  }
  func @S0(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = sitofp %c1_i32 : i32 to f64
    %1 = affine.load %arg1[0] : memref<1xf64>
    %2 = affine.load %arg1[0] : memref<1xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    %5 = affine.load %arg0[0] : memref<1xf64>
    %6 = mulf %4, %5 : f64
    affine.store %6, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: memref<2000xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg3[%arg4 - %arg2 - 1] : memref<2000xf64>
    %2 = affine.load %arg1[%arg2] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<1xf64>, %arg2: memref<1xf64>, %arg3: memref<1xf64>, %arg4: memref<2000xf64>, %arg5: index) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = affine.load %arg4[%arg5] : memref<2000xf64>
    %1 = affine.load %arg3[0] : memref<1xf64>
    %2 = addf %0, %1 : f64
    %3 = negf %2 : f64
    %4 = sitofp %c1_i32 : i32 to f64
    %5 = affine.load %arg2[0] : memref<1xf64>
    %6 = affine.load %arg2[0] : memref<1xf64>
    %7 = mulf %5, %6 : f64
    %8 = subf %4, %7 : f64
    %9 = affine.load %arg1[0] : memref<1xf64>
    %10 = mulf %8, %9 : f64
    %11 = divf %3, %10 : f64
    affine.store %11, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: memref<1xf64>, %arg2: memref<1xf64>, %arg3: memref<2000xf64>, %arg4: index) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = affine.load %arg3[%arg4] : memref<2000xf64>
    %1 = affine.load %arg2[0] : memref<1xf64>
    %2 = addf %0, %1 : f64
    %3 = negf %2 : f64
    %4 = sitofp %c1_i32 : i32 to f64
    %5 = affine.load %arg0[0] : memref<1xf64>
    %6 = affine.load %arg0[0] : memref<1xf64>
    %7 = mulf %5, %6 : f64
    %8 = subf %4, %7 : f64
    %9 = affine.load %arg1[0] : memref<1xf64>
    %10 = mulf %8, %9 : f64
    %11 = divf %3, %10 : f64
    affine.store %11, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: index, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: index, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[%arg0] : memref<2000xf64>
    %1 = affine.load %arg4[0] : memref<1xf64>
    %2 = affine.load %arg2[%arg3 - %arg0 - 1] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg1[%arg0] : memref<2000xf64>
    return
  }
  func @S6(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[%arg1] : memref<2000xf64>
    affine.store %0, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @S7(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<1xf64>, %arg3: memref<1xf64>, %arg4: memref<1xf64>, %arg5: memref<2000xf64>) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = affine.load %arg5[%arg1] : memref<2000xf64>
    %1 = affine.load %arg4[0] : memref<1xf64>
    %2 = addf %0, %1 : f64
    %3 = negf %2 : f64
    %4 = sitofp %c1_i32 : i32 to f64
    %5 = affine.load %arg3[0] : memref<1xf64>
    %6 = affine.load %arg3[0] : memref<1xf64>
    %7 = mulf %5, %6 : f64
    %8 = subf %4, %7 : f64
    %9 = affine.load %arg2[0] : memref<1xf64>
    %10 = mulf %8, %9 : f64
    %11 = divf %3, %10 : f64
    affine.store %11, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @kernel_durbin_new(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
    %c0 = constant 0 : index
    %0 = alloca() : memref<2000xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = alloca() : memref<1xf64>
    %5 = index_cast %arg0 : i32 to index
    affine.for %arg3 = 1 to %5 {
      call @S1(%2) : (memref<1xf64>) -> ()
      call @S2(%2, %arg2, %arg3, %arg1, %c0) : (memref<1xf64>, memref<2000xf64>, index, memref<2000xf64>, index) -> ()
      call @S0(%3, %4) : (memref<1xf64>, memref<1xf64>) -> ()
      affine.for %arg4 = 1 to #map0(%arg3) {
        call @S2(%2, %arg2, %arg3, %arg1, %arg4) : (memref<1xf64>, memref<2000xf64>, index, memref<2000xf64>, index) -> ()
      }
      call @S3(%1, %3, %4, %2, %arg1, %arg3) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S5(%arg3, %0, %arg2, %c0, %1) : (index, memref<2000xf64>, memref<2000xf64>, index, memref<1xf64>) -> ()
      call @S4(%4, %3, %2, %arg1, %arg3) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S7(%arg2, %arg3, %3, %4, %2, %arg1) : (memref<2000xf64>, index, memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>) -> ()
      affine.for %arg4 = #map1(%arg3) to #map2(%arg3) {
        %6 = affine.apply #map3(%arg3, %arg4)
        call @S5(%arg3, %0, %arg2, %6, %1) : (index, memref<2000xf64>, memref<2000xf64>, index, memref<1xf64>) -> ()
      }
      affine.for %arg4 = #map2(%arg3) to #map4(%arg3) {
        call @S6(%arg2, %arg3, %0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
    }
    return
  }
}

