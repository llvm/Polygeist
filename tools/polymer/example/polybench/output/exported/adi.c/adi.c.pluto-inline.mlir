#map0 = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>
#map3 = affine_map<(d0) -> (1, d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 32)>
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
  llvm.mlir.global internal constant @str3("u\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1000_i32 = constant 1000 : i32
    %c500_i32 = constant 500 : i32
    %c42_i32 = constant 42 : i32
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %c1000 = constant 1000 : index
    %c0 = constant 0 : index
    %0 = alloc() : memref<1000x1000xf64>
    %1 = alloc() : memref<1000x1000xf64>
    %2 = alloc() : memref<1000x1000xf64>
    %3 = alloc() : memref<1000x1000xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb5
    %5 = cmpi "slt", %4, %c1000_i32 : i32
    %6 = index_cast %4 : i32 to index
    cond_br %5, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %7 = get_global_memref @polybench_t_start : memref<1xf64>
    %8 = call @rtclock() : () -> f64
    store %8, %7[%c0] : memref<1xf64>
    %9 = alloca() : memref<1xf64>
    %10 = sitofp %c500_i32 : i32 to f64
    %11 = divf %cst_0, %10 : f64
    %12 = sitofp %c1000_i32 : i32 to f64
    %13 = divf %cst_0, %12 : f64
    %14 = mulf %13, %13 : f64
    %15 = mulf %cst, %11 : f64
    %16 = divf %15, %14 : f64
    %17 = negf %16 : f64
    %18 = divf %17, %cst : f64
    affine.store %18, %9[0] : memref<1xf64>
    %19 = alloca() : memref<1xf64>
    %20 = sitofp %c500_i32 : i32 to f64
    %21 = divf %cst_0, %20 : f64
    %22 = mulf %cst, %21 : f64
    %23 = sitofp %c1000_i32 : i32 to f64
    %24 = divf %cst_0, %23 : f64
    %25 = mulf %24, %24 : f64
    %26 = divf %22, %25 : f64
    %27 = addf %cst_0, %26 : f64
    affine.store %27, %19[0] : memref<1xf64>
    %28 = alloca() : memref<1xf64>
    %29 = sitofp %c500_i32 : i32 to f64
    %30 = divf %cst_0, %29 : f64
    %31 = mulf %cst_0, %30 : f64
    %32 = sitofp %c1000_i32 : i32 to f64
    %33 = divf %cst_0, %32 : f64
    %34 = mulf %33, %33 : f64
    %35 = divf %31, %34 : f64
    %36 = negf %35 : f64
    %37 = divf %36, %cst : f64
    affine.store %37, %28[0] : memref<1xf64>
    %38 = alloca() : memref<1xf64>
    %39 = sitofp %c500_i32 : i32 to f64
    %40 = divf %cst_0, %39 : f64
    %41 = mulf %cst_0, %40 : f64
    %42 = sitofp %c1000_i32 : i32 to f64
    %43 = divf %cst_0, %42 : f64
    %44 = mulf %43, %43 : f64
    %45 = divf %41, %44 : f64
    %46 = addf %cst_0, %45 : f64
    affine.store %46, %38[0] : memref<1xf64>
    affine.for %arg2 = 1 to 501 {
      affine.for %arg3 = 1 to 999 {
        call @S4(%1, %arg3) : (memref<1000x1000xf64>, index) -> ()
        call @S5(%2, %arg3) : (memref<1000x1000xf64>, index) -> ()
        call @S6(%3, %arg3, %1) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        affine.for %arg4 = 1 to 999 {
          call @S7(%2, %arg3, %arg4, %19, %9) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>) -> ()
          call @S8(%3, %arg3, %arg4, %19, %2, %9, %0, %28) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>) -> ()
        }
        call @S9(%1, %arg3, %c1000) : (memref<1000x1000xf64>, index, index) -> ()
        affine.for %arg4 = 1 to 999 {
          call @S10(%1, %arg4, %arg3, %c1000, %3, %2) : (memref<1000x1000xf64>, index, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
        }
      }
      affine.for %arg3 = 1 to 999 {
        call @S11(%0, %arg3) : (memref<1000x1000xf64>, index) -> ()
        call @S12(%2, %arg3) : (memref<1000x1000xf64>, index) -> ()
        call @S13(%3, %arg3, %0) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        affine.for %arg4 = 1 to 999 {
          call @S14(%2, %arg3, %arg4, %38, %28) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>) -> ()
          call @S15(%3, %arg3, %arg4, %38, %2, %28, %1, %9) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>) -> ()
        }
        call @S16(%0, %arg3, %c1000) : (memref<1000x1000xf64>, index, index) -> ()
        affine.for %arg4 = 1 to 999 {
          call @S17(%0, %arg3, %arg4, %c1000, %3, %2) : (memref<1000x1000xf64>, index, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
        }
      }
    }
    %47 = get_global_memref @polybench_t_end : memref<1xf64>
    %48 = call @rtclock() : () -> f64
    store %48, %47[%c0] : memref<1xf64>
    call @polybench_timer_print() : () -> ()
    %49 = cmpi "sgt", %arg0, %c42_i32 : i32
    %50 = scf.if %49 -> (i1) {
      %61 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
      %62 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %63 = llvm.mlir.constant(0 : index) : !llvm.i64
      %64 = llvm.getelementptr %62[%63, %63] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %65 = llvm.call @strcmp(%61, %64) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
      %66 = llvm.mlir.cast %65 : !llvm.i32 to i32
      %67 = trunci %66 : i32 to i1
      %68 = xor %67, %true : i1
      scf.yield %68 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %50 {
      call @print_array(%c1000_i32, %0) : (i32, memref<1000x1000xf64>) -> ()
    }
    return %c0_i32 : i32
  ^bb3(%51: i32):  // 2 preds: ^bb1, ^bb4
    %52 = cmpi "slt", %51, %c1000_i32 : i32
    %53 = index_cast %51 : i32 to index
    cond_br %52, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %54 = addi %4, %c1000_i32 : i32
    %55 = subi %54, %51 : i32
    %56 = sitofp %55 : i32 to f64
    %57 = sitofp %c1000_i32 : i32 to f64
    %58 = divf %56, %57 : f64
    store %58, %0[%6, %53] : memref<1000x1000xf64>
    %59 = addi %51, %c1_i32 : i32
    br ^bb3(%59 : i32)
  ^bb5:  // pred: ^bb3
    %60 = addi %4, %c1_i32 : i32
    br ^bb1(%60 : i32)
  }
  func @init_array(%arg0: i32, %arg1: memref<1000x1000xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb5
    %1 = cmpi "slt", %0, %arg0 : i32
    %2 = index_cast %0 : i32 to index
    cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    return
  ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb4
    %4 = cmpi "slt", %3, %arg0 : i32
    %5 = index_cast %3 : i32 to index
    cond_br %4, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %6 = addi %0, %arg0 : i32
    %7 = subi %6, %3 : i32
    %8 = sitofp %7 : i32 to f64
    %9 = sitofp %arg0 : i32 to f64
    %10 = divf %8, %9 : f64
    store %10, %arg1[%2, %5] : memref<1000x1000xf64>
    %11 = addi %3, %c1_i32 : i32
    br ^bb3(%11 : i32)
  ^bb5:  // pred: ^bb3
    %12 = addi %0, %c1_i32 : i32
    br ^bb1(%12 : i32)
  }
  func @polybench_timer_start() {
    %c0 = constant 0 : index
    %0 = get_global_memref @polybench_t_start : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = index_cast %arg1 : i32 to index
    %1 = alloca() : memref<1xf64>
    %2 = sitofp %arg0 : i32 to f64
    %3 = divf %cst_0, %2 : f64
    %4 = sitofp %arg1 : i32 to f64
    %5 = divf %cst_0, %4 : f64
    %6 = mulf %5, %5 : f64
    %7 = mulf %cst, %3 : f64
    %8 = divf %7, %6 : f64
    %9 = negf %8 : f64
    %10 = divf %9, %cst : f64
    affine.store %10, %1[0] : memref<1xf64>
    %11 = alloca() : memref<1xf64>
    %12 = sitofp %arg0 : i32 to f64
    %13 = divf %cst_0, %12 : f64
    %14 = mulf %cst, %13 : f64
    %15 = sitofp %arg1 : i32 to f64
    %16 = divf %cst_0, %15 : f64
    %17 = mulf %16, %16 : f64
    %18 = divf %14, %17 : f64
    %19 = addf %cst_0, %18 : f64
    affine.store %19, %11[0] : memref<1xf64>
    %20 = alloca() : memref<1xf64>
    %21 = sitofp %arg0 : i32 to f64
    %22 = divf %cst_0, %21 : f64
    %23 = mulf %cst_0, %22 : f64
    %24 = sitofp %arg1 : i32 to f64
    %25 = divf %cst_0, %24 : f64
    %26 = mulf %25, %25 : f64
    %27 = divf %23, %26 : f64
    %28 = negf %27 : f64
    %29 = divf %28, %cst : f64
    affine.store %29, %20[0] : memref<1xf64>
    %30 = alloca() : memref<1xf64>
    %31 = sitofp %arg0 : i32 to f64
    %32 = divf %cst_0, %31 : f64
    %33 = mulf %cst_0, %32 : f64
    %34 = sitofp %arg1 : i32 to f64
    %35 = divf %cst_0, %34 : f64
    %36 = mulf %35, %35 : f64
    %37 = divf %33, %36 : f64
    %38 = addf %cst_0, %37 : f64
    affine.store %38, %30[0] : memref<1xf64>
    %39 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 1 to #map0()[%39] {
      affine.for %arg7 = 1 to #map1()[%0] {
        call @S4(%arg3, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S5(%arg4, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S6(%arg5, %arg7, %arg3) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        affine.for %arg8 = 1 to #map1()[%0] {
          call @S7(%arg4, %arg7, %arg8, %11, %1) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>) -> ()
          call @S8(%arg5, %arg7, %arg8, %11, %arg4, %1, %arg2, %20) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>) -> ()
        }
        call @S9(%arg3, %arg7, %0) : (memref<1000x1000xf64>, index, index) -> ()
        affine.for %arg8 = 1 to #map1()[%0] {
          call @S10(%arg3, %arg8, %arg7, %0, %arg5, %arg4) : (memref<1000x1000xf64>, index, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
        }
      }
      affine.for %arg7 = 1 to #map1()[%0] {
        call @S11(%arg2, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S12(%arg4, %arg7) : (memref<1000x1000xf64>, index) -> ()
        call @S13(%arg5, %arg7, %arg2) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        affine.for %arg8 = 1 to #map1()[%0] {
          call @S14(%arg4, %arg7, %arg8, %30, %20) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>) -> ()
          call @S15(%arg5, %arg7, %arg8, %30, %arg4, %20, %arg3, %1) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>) -> ()
        }
        call @S16(%arg2, %arg7, %0) : (memref<1000x1000xf64>, index, index) -> ()
        affine.for %arg8 = 1 to #map1()[%0] {
          call @S17(%arg2, %arg7, %arg8, %0, %arg5, %arg4) : (memref<1000x1000xf64>, index, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
        }
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
  func @print_array(%arg0: i32, %arg1: memref<1000x1000xf64>) {
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
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %16 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %17 = llvm.load %16 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %18 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %19 = llvm.getelementptr %18[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %20 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = llvm.call @fprintf(%17, %19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %23 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %24 = llvm.load %23 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %25 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %26 = llvm.getelementptr %25[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %27 = llvm.call @fprintf(%24, %26) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  ^bb3(%28: i32):  // 2 preds: ^bb1, ^bb4
    %29 = cmpi "slt", %28, %arg0 : i32
    %30 = index_cast %28 : i32 to index
    cond_br %29, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %31 = muli %13, %arg0 : i32
    %32 = addi %31, %28 : i32
    %33 = remi_signed %32, %c20_i32 : i32
    %34 = cmpi "eq", %33, %c0_i32 : i32
    scf.if %34 {
      %44 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %45 = llvm.load %44 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %46 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
      %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg1[%15, %30] : memref<1000x1000xf64>
    %40 = llvm.mlir.cast %39 : f64 to !llvm.double
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %42 = addi %28, %c1_i32 : i32
    br ^bb3(%42 : i32)
  ^bb5:  // pred: ^bb3
    %43 = addi %13, %c1_i32 : i32
    br ^bb1(%43 : i32)
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
  func private @S4(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    affine.store %cst, %arg0[0, %arg1] : memref<1000x1000xf64>
    return
  }
  func private @S5(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, 0] : memref<1000x1000xf64>
    return
  }
  func private @S6(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: memref<1000x1000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0, %arg1] : memref<1000x1000xf64>
    affine.store %0, %arg0[%arg1, 0] : memref<1000x1000xf64>
    return
  }
  func private @S7(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[0] : memref<1xf64>
    %1 = negf %0 : f64
    %2 = affine.load %arg0[%arg1, %arg2 - 1] : memref<1000x1000xf64>
    %3 = mulf %0, %2 : f64
    %4 = affine.load %arg3[0] : memref<1xf64>
    %5 = addf %3, %4 : f64
    %6 = divf %1, %5 : f64
    affine.store %6, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func private @S8(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1xf64>, %arg6: memref<1000x1000xf64>, %arg7: memref<1xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = affine.load %arg6[%arg2, %arg1 - 1] : memref<1000x1000xf64>
    %1 = affine.load %arg6[%arg2, %arg1] : memref<1000x1000xf64>
    %2 = affine.load %arg7[0] : memref<1xf64>
    %3 = mulf %cst, %2 : f64
    %4 = addf %cst_0, %3 : f64
    %5 = mulf %4, %1 : f64
    %6 = negf %2 : f64
    %7 = mulf %6, %0 : f64
    %8 = addf %7, %5 : f64
    %9 = affine.load %arg6[%arg2, %arg1 + 1] : memref<1000x1000xf64>
    %10 = mulf %2, %9 : f64
    %11 = subf %8, %10 : f64
    %12 = affine.load %arg0[%arg1, %arg2 - 1] : memref<1000x1000xf64>
    %13 = affine.load %arg5[0] : memref<1xf64>
    %14 = mulf %13, %12 : f64
    %15 = subf %11, %14 : f64
    %16 = affine.load %arg4[%arg1, %arg2 - 1] : memref<1000x1000xf64>
    %17 = mulf %13, %16 : f64
    %18 = affine.load %arg3[0] : memref<1xf64>
    %19 = addf %17, %18 : f64
    %20 = divf %15, %19 : f64
    affine.store %20, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func private @S9(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg2) - 1, %arg1] : memref<1000x1000xf64>
    return
  }
  func private @S10(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[%arg2, -%arg1 + symbol(%arg3) - 1] : memref<1000x1000xf64>
    %1 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2] : memref<1000x1000xf64>
    %2 = mulf %0, %1 : f64
    %3 = affine.load %arg4[%arg2, -%arg1 + symbol(%arg3) - 1] : memref<1000x1000xf64>
    %4 = addf %2, %3 : f64
    affine.store %4, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<1000x1000xf64>
    return
  }
  func private @S11(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, 0] : memref<1000x1000xf64>
    return
  }
  func private @S12(%arg0: memref<1000x1000xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, 0] : memref<1000x1000xf64>
    return
  }
  func private @S13(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: memref<1000x1000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[%arg1, 0] : memref<1000x1000xf64>
    affine.store %0, %arg0[%arg1, 0] : memref<1000x1000xf64>
    return
  }
  func private @S14(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[0] : memref<1xf64>
    %1 = negf %0 : f64
    %2 = affine.load %arg0[%arg1, %arg2 - 1] : memref<1000x1000xf64>
    %3 = mulf %0, %2 : f64
    %4 = affine.load %arg3[0] : memref<1xf64>
    %5 = addf %3, %4 : f64
    %6 = divf %1, %5 : f64
    affine.store %6, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func private @S15(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1xf64>, %arg6: memref<1000x1000xf64>, %arg7: memref<1xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = affine.load %arg6[%arg1 - 1, %arg2] : memref<1000x1000xf64>
    %1 = affine.load %arg6[%arg1, %arg2] : memref<1000x1000xf64>
    %2 = affine.load %arg7[0] : memref<1xf64>
    %3 = mulf %cst, %2 : f64
    %4 = addf %cst_0, %3 : f64
    %5 = mulf %4, %1 : f64
    %6 = negf %2 : f64
    %7 = mulf %6, %0 : f64
    %8 = addf %7, %5 : f64
    %9 = affine.load %arg6[%arg1 + 1, %arg2] : memref<1000x1000xf64>
    %10 = mulf %2, %9 : f64
    %11 = subf %8, %10 : f64
    %12 = affine.load %arg0[%arg1, %arg2 - 1] : memref<1000x1000xf64>
    %13 = affine.load %arg5[0] : memref<1xf64>
    %14 = mulf %13, %12 : f64
    %15 = subf %11, %14 : f64
    %16 = affine.load %arg4[%arg1, %arg2 - 1] : memref<1000x1000xf64>
    %17 = mulf %13, %16 : f64
    %18 = affine.load %arg3[0] : memref<1xf64>
    %19 = addf %17, %18 : f64
    %20 = divf %15, %19 : f64
    affine.store %20, %arg0[%arg1, %arg2] : memref<1000x1000xf64>
    return
  }
  func private @S16(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, symbol(%arg2) - 1] : memref<1000x1000xf64>
    return
  }
  func private @S17(%arg0: memref<1000x1000xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[%arg1, -%arg2 + symbol(%arg3) - 1] : memref<1000x1000xf64>
    %1 = affine.load %arg0[%arg1, -%arg2 + symbol(%arg3)] : memref<1000x1000xf64>
    %2 = mulf %0, %1 : f64
    %3 = affine.load %arg4[%arg1, -%arg2 + symbol(%arg3) - 1] : memref<1000x1000xf64>
    %4 = addf %2, %3 : f64
    affine.store %4, %arg0[%arg1, -%arg2 + symbol(%arg3) - 1] : memref<1000x1000xf64>
    return
  }
  func @kernel_adi_new(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = index_cast %arg1 : i32 to index
    %5 = index_cast %arg0 : i32 to index
    %6 = sitofp %arg0 : i32 to f64
    %7 = divf %cst_0, %6 : f64
    %8 = sitofp %arg1 : i32 to f64
    %9 = divf %cst_0, %8 : f64
    %10 = mulf %9, %9 : f64
    %11 = mulf %cst, %7 : f64
    %12 = divf %11, %10 : f64
    %13 = negf %12 : f64
    %14 = divf %13, %cst : f64
    affine.store %14, %2[0] : memref<1xf64>
    %15 = sitofp %arg0 : i32 to f64
    %16 = divf %cst_0, %15 : f64
    %17 = mulf %cst, %16 : f64
    %18 = sitofp %arg1 : i32 to f64
    %19 = divf %cst_0, %18 : f64
    %20 = mulf %19, %19 : f64
    %21 = divf %17, %20 : f64
    %22 = addf %cst_0, %21 : f64
    affine.store %22, %1[0] : memref<1xf64>
    affine.for %arg6 = 1 to #map0()[%5] {
      affine.for %arg7 = 0 to #map2()[%4] {
        affine.for %arg8 = max #map3(%arg7) to min #map4(%arg7)[%4] {
          call @S4(%arg3, %arg6) : (memref<1000x1000xf64>, index) -> ()
          call @S5(%arg4, %arg6) : (memref<1000x1000xf64>, index) -> ()
          call @S6(%arg5, %arg6, %arg3) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        }
      }
      affine.for %arg7 = 0 to #map2()[%4] {
        affine.for %arg8 = 0 to #map2()[%4] {
          affine.for %arg9 = max #map3(%arg7) to min #map4(%arg7)[%4] {
            affine.for %arg10 = max #map3(%arg8) to min #map4(%arg8)[%4] {
              call @S10(%arg3, %arg6, %arg9, %4, %arg5, %arg4) : (memref<1000x1000xf64>, index, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map2()[%4] {
        affine.for %arg8 = max #map3(%arg7) to min #map4(%arg7)[%4] {
          call @S9(%arg3, %arg6, %4) : (memref<1000x1000xf64>, index, index) -> ()
        }
      }
      affine.for %arg7 = 0 to #map2()[%4] {
        affine.for %arg8 = 0 to #map2()[%4] {
          affine.for %arg9 = max #map3(%arg7) to min #map4(%arg7)[%4] {
            affine.for %arg10 = max #map3(%arg8) to min #map4(%arg8)[%4] {
              call @S7(%arg4, %arg6, %arg9, %1, %2) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>) -> ()
            }
          }
        }
        affine.for %arg8 = 0 to #map2()[%4] {
          affine.for %arg9 = max #map3(%arg7) to min #map4(%arg7)[%4] {
            affine.for %arg10 = max #map3(%arg8) to min #map4(%arg8)[%4] {
              call @S8(%arg5, %arg6, %arg9, %1, %arg4, %2, %arg2, %0) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>) -> ()
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map2()[%4] {
        affine.for %arg8 = max #map3(%arg7) to min #map4(%arg7)[%4] {
          call @S16(%arg2, %arg6, %4) : (memref<1000x1000xf64>, index, index) -> ()
          call @S11(%arg2, %arg6) : (memref<1000x1000xf64>, index) -> ()
          call @S12(%arg4, %arg6) : (memref<1000x1000xf64>, index) -> ()
          call @S13(%arg5, %arg6, %arg2) : (memref<1000x1000xf64>, index, memref<1000x1000xf64>) -> ()
        }
      }
      affine.for %arg7 = 0 to #map2()[%4] {
        affine.for %arg8 = 0 to #map2()[%4] {
          affine.for %arg9 = max #map3(%arg7) to min #map4(%arg7)[%4] {
            affine.for %arg10 = max #map3(%arg8) to min #map4(%arg8)[%4] {
              call @S14(%arg4, %arg6, %arg9, %3, %0) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1xf64>) -> ()
            }
          }
        }
        affine.for %arg8 = 0 to #map2()[%4] {
          affine.for %arg9 = max #map3(%arg7) to min #map4(%arg7)[%4] {
            affine.for %arg10 = max #map3(%arg8) to min #map4(%arg8)[%4] {
              call @S15(%arg5, %arg6, %arg9, %3, %arg4, %0, %arg3, %2) : (memref<1000x1000xf64>, index, index, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>, memref<1000x1000xf64>, memref<1xf64>) -> ()
            }
          }
        }
        affine.for %arg8 = 0 to #map2()[%4] {
          affine.for %arg9 = max #map3(%arg7) to min #map4(%arg7)[%4] {
            affine.for %arg10 = max #map3(%arg8) to min #map4(%arg8)[%4] {
              call @S17(%arg2, %arg6, %arg9, %4, %arg5, %arg4) : (memref<1000x1000xf64>, index, index, index, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
            }
          }
        }
      }
    }
    %23 = sitofp %arg0 : i32 to f64
    %24 = divf %cst_0, %23 : f64
    %25 = mulf %cst_0, %24 : f64
    %26 = sitofp %arg1 : i32 to f64
    %27 = divf %cst_0, %26 : f64
    %28 = mulf %27, %27 : f64
    %29 = divf %25, %28 : f64
    %30 = negf %29 : f64
    %31 = divf %30, %cst : f64
    affine.store %31, %0[0] : memref<1xf64>
    %32 = sitofp %arg0 : i32 to f64
    %33 = divf %cst_0, %32 : f64
    %34 = mulf %cst_0, %33 : f64
    %35 = sitofp %arg1 : i32 to f64
    %36 = divf %cst_0, %35 : f64
    %37 = mulf %36, %36 : f64
    %38 = divf %34, %37 : f64
    %39 = addf %cst_0, %38 : f64
    affine.store %39, %3[0] : memref<1xf64>
    return
  }
}

