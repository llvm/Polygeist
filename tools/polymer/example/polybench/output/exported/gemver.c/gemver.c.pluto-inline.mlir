#map0 = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>
#map1 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map2 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32)>
#map6 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map7 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
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
  llvm.mlir.global internal constant @str3("w\00")
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
    %c0 = constant 0 : index
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloc() : memref<2000x2000xf64>
    %3 = alloc() : memref<2000xf64>
    %4 = alloc() : memref<2000xf64>
    %5 = alloc() : memref<2000xf64>
    %6 = alloc() : memref<2000xf64>
    %7 = alloc() : memref<2000xf64>
    %8 = alloc() : memref<2000xf64>
    %9 = alloc() : memref<2000xf64>
    %10 = alloc() : memref<2000xf64>
    store %cst, %0[%c0] : memref<1xf64>
    store %cst_0, %1[%c0] : memref<1xf64>
    %11 = sitofp %c2000_i32 : i32 to f64
    br ^bb1(%c0_i32 : i32)
  ^bb1(%12: i32):  // 2 preds: ^bb0, ^bb4
    %13 = cmpi "slt", %12, %c2000_i32 : i32
    %14 = index_cast %12 : i32 to index
    cond_br %13, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %15 = sitofp %12 : i32 to f64
    store %15, %3[%14] : memref<2000xf64>
    %16 = addi %12, %c1_i32 : i32
    %17 = sitofp %16 : i32 to f64
    %18 = divf %17, %11 : f64
    %19 = divf %18, %cst_1 : f64
    store %19, %5[%14] : memref<2000xf64>
    %20 = divf %18, %cst_2 : f64
    store %20, %4[%14] : memref<2000xf64>
    %21 = divf %18, %cst_3 : f64
    store %21, %6[%14] : memref<2000xf64>
    %22 = divf %18, %cst_4 : f64
    store %22, %9[%14] : memref<2000xf64>
    %23 = divf %18, %cst_5 : f64
    store %23, %10[%14] : memref<2000xf64>
    store %cst_6, %8[%14] : memref<2000xf64>
    store %cst_6, %7[%14] : memref<2000xf64>
    br ^bb4(%c0_i32 : i32)
  ^bb3:  // pred: ^bb1
    %24 = get_global_memref @polybench_t_start : memref<1xf64>
    %25 = call @rtclock() : () -> f64
    store %25, %24[%c0] : memref<1xf64>
    %26 = load %0[%c0] : memref<1xf64>
    %27 = load %1[%c0] : memref<1xf64>
    affine.for %arg2 = 0 to 2000 {
      affine.for %arg3 = 0 to 2000 {
        call @S0(%2, %arg2, %arg3, %6, %5, %4, %3) : (memref<2000x2000xf64>, index, index, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
      }
    }
    affine.for %arg2 = 0 to 2000 {
      affine.for %arg3 = 0 to 2000 {
        call @S1(%8, %arg2, %9, %arg3, %27, %2) : (memref<2000xf64>, index, memref<2000xf64>, index, f64, memref<2000x2000xf64>) -> ()
      }
    }
    affine.for %arg2 = 0 to 2000 {
      call @S2(%8, %arg2, %10) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
    }
    affine.for %arg2 = 0 to 2000 {
      affine.for %arg3 = 0 to 2000 {
        call @S3(%7, %arg2, %8, %arg3, %26, %2) : (memref<2000xf64>, index, memref<2000xf64>, index, f64, memref<2000x2000xf64>) -> ()
      }
    }
    %28 = get_global_memref @polybench_t_end : memref<1xf64>
    %29 = call @rtclock() : () -> f64
    store %29, %28[%c0] : memref<1xf64>
    call @polybench_timer_print() : () -> ()
    %30 = cmpi "sgt", %arg0, %c42_i32 : i32
    %31 = scf.if %30 -> (i1) {
      %40 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
      %41 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %42 = llvm.mlir.constant(0 : index) : !llvm.i64
      %43 = llvm.getelementptr %41[%42, %42] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %44 = llvm.call @strcmp(%40, %43) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
      %45 = llvm.mlir.cast %44 : !llvm.i32 to i32
      %46 = trunci %45 : i32 to i1
      %47 = xor %46, %true : i1
      scf.yield %47 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %31 {
      call @print_array(%c2000_i32, %7) : (i32, memref<2000xf64>) -> ()
    }
    return %c0_i32 : i32
  ^bb4(%32: i32):  // 2 preds: ^bb2, ^bb5
    %33 = cmpi "slt", %32, %c2000_i32 : i32
    %34 = index_cast %32 : i32 to index
    cond_br %33, ^bb5, ^bb1(%16 : i32)
  ^bb5:  // pred: ^bb4
    %35 = muli %12, %32 : i32
    %36 = remi_signed %35, %c2000_i32 : i32
    %37 = sitofp %36 : i32 to f64
    %38 = divf %37, %11 : f64
    store %38, %2[%14, %34] : memref<2000x2000xf64>
    %39 = addi %32, %c1_i32 : i32
    br ^bb4(%39 : i32)
  }
  func @init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
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
    store %4, %arg4[%3] : memref<2000xf64>
    %5 = addi %1, %c1_i32 : i32
    %6 = sitofp %5 : i32 to f64
    %7 = divf %6, %0 : f64
    %8 = divf %7, %cst_1 : f64
    store %8, %arg6[%3] : memref<2000xf64>
    %9 = divf %7, %cst_2 : f64
    store %9, %arg5[%3] : memref<2000xf64>
    %10 = divf %7, %cst_3 : f64
    store %10, %arg7[%3] : memref<2000xf64>
    %11 = divf %7, %cst_4 : f64
    store %11, %arg10[%3] : memref<2000xf64>
    %12 = divf %7, %cst_5 : f64
    store %12, %arg11[%3] : memref<2000xf64>
    store %cst_6, %arg9[%3] : memref<2000xf64>
    store %cst_6, %arg8[%3] : memref<2000xf64>
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
    store %19, %arg3[%3, %15] : memref<2000x2000xf64>
    %20 = addi %13, %c1_i32 : i32
    br ^bb4(%20 : i32)
  }
  func @polybench_timer_start() {
    %c0 = constant 0 : index
    %0 = get_global_memref @polybench_t_start : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        call @S0(%arg3, %arg12, %arg13, %arg7, %arg6, %arg5, %arg4) : (memref<2000x2000xf64>, index, index, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        call @S1(%arg9, %arg12, %arg10, %arg13, %arg2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>, index, f64, memref<2000x2000xf64>) -> ()
      }
    }
    affine.for %arg12 = 0 to %0 {
      call @S2(%arg9, %arg12, %arg11) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        call @S3(%arg8, %arg12, %arg9, %arg13, %arg1, %arg3) : (memref<2000xf64>, index, memref<2000xf64>, index, f64, memref<2000x2000xf64>) -> ()
      }
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
  func private @S0(%arg0: memref<2000x2000xf64>, %arg1: index, %arg2: index, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    %1 = affine.load %arg6[%arg1] : memref<2000xf64>
    %2 = affine.load %arg5[%arg2] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    %5 = affine.load %arg4[%arg1] : memref<2000xf64>
    %6 = affine.load %arg3[%arg2] : memref<2000xf64>
    %7 = mulf %5, %6 : f64
    %8 = addf %4, %7 : f64
    affine.store %8, %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    return
  }
  func private @S1(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>, %arg3: index, %arg4: f64, %arg5: memref<2000x2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<2000xf64>
    %1 = affine.load %arg5[%arg3, %arg1] : memref<2000x2000xf64>
    %2 = mulf %arg4, %1 : f64
    %3 = affine.load %arg2[%arg3] : memref<2000xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func private @S2(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<2000xf64>
    %1 = affine.load %arg2[%arg1] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    affine.store %2, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func private @S3(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>, %arg3: index, %arg4: f64, %arg5: memref<2000x2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<2000xf64>
    %1 = affine.load %arg5[%arg1, %arg3] : memref<2000x2000xf64>
    %2 = mulf %arg4, %1 : f64
    %3 = affine.load %arg2[%arg3] : memref<2000xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @kernel_gemver_new(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to #map0()[%0] {
      affine.for %arg13 = max #map1(%arg12)[%0] to min #map2(%arg12)[%0] {
        affine.for %arg14 = #map3(%arg13) to min #map4(%arg13)[%0] {
          affine.for %arg15 = #map5(%arg12, %arg13) to min #map6(%arg12, %arg13)[%0] {
            call @S0(%arg3, %arg15, %arg14, %arg7, %arg6, %arg5, %arg4) : (memref<2000x2000xf64>, index, index, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
            call @S1(%arg9, %arg14, %arg10, %arg15, %arg2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>, index, f64, memref<2000x2000xf64>) -> ()
          }
        }
      }
    }
    affine.for %arg12 = 0 to #map7()[%0] {
      affine.for %arg13 = #map3(%arg12) to min #map4(%arg12)[%0] {
        call @S2(%arg9, %arg13, %arg11) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
    }
    affine.for %arg12 = 0 to #map7()[%0] {
      affine.for %arg13 = 0 to #map7()[%0] {
        affine.for %arg14 = #map3(%arg12) to min #map4(%arg12)[%0] {
          affine.for %arg15 = #map3(%arg13) to min #map4(%arg13)[%0] {
            call @S3(%arg8, %arg14, %arg9, %arg15, %arg1, %arg3) : (memref<2000xf64>, index, memref<2000xf64>, index, f64, memref<2000x2000xf64>) -> ()
          }
        }
      }
    }
    return
  }
}

