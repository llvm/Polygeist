#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (-d0 + 2000)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0)>
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
  llvm.mlir.global internal constant @str3("x\00")
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
    %cst = constant 2.000000e+00 : f64
    %c4_i32 = constant 4 : i32
    %c1_i32 = constant 1 : i32
    %c1 = constant 1 : index
    %c2000 = constant 2000 : index
    %c0 = constant 0 : index
    %0 = alloc() : memref<2000x2000xf64>
    %1 = alloc() : memref<2000xf64>
    %2 = alloc() : memref<2000xf64>
    %3 = alloc() : memref<2000xf64>
    %4 = sitofp %c2000_i32 : i32 to f64
    br ^bb1(%c0_i32 : i32)
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb2
    %6 = cmpi "slt", %5, %c2000_i32 : i32
    %7 = index_cast %5 : i32 to index
    cond_br %6, ^bb2, ^bb3(%c0_i32 : i32)
  ^bb2:  // pred: ^bb1
    %8 = sitofp %c0_i32 : i32 to f64
    store %8, %2[%7] : memref<2000xf64>
    store %8, %3[%7] : memref<2000xf64>
    %9 = addi %5, %c1_i32 : i32
    %10 = sitofp %9 : i32 to f64
    %11 = divf %10, %4 : f64
    %12 = divf %11, %cst : f64
    %13 = sitofp %c4_i32 : i32 to f64
    %14 = addf %12, %13 : f64
    store %14, %1[%7] : memref<2000xf64>
    br ^bb1(%9 : i32)
  ^bb3(%15: i32):  // 2 preds: ^bb1, ^bb10
    %16 = cmpi "slt", %15, %c2000_i32 : i32
    %17 = index_cast %15 : i32 to index
    cond_br %16, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %18 = alloc() : memref<2000x2000xf64>
    br ^bb11(%c0_i32 : i32)
  ^bb5(%19: i32):  // 2 preds: ^bb3, ^bb6
    %20 = cmpi "sle", %19, %15 : i32
    %21 = index_cast %19 : i32 to index
    cond_br %20, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %22 = subi %c0_i32, %19 : i32
    %23 = remi_signed %22, %c2000_i32 : i32
    %24 = sitofp %23 : i32 to f64
    %25 = divf %24, %4 : f64
    %26 = sitofp %c1_i32 : i32 to f64
    %27 = addf %25, %26 : f64
    store %27, %0[%17, %21] : memref<2000x2000xf64>
    %28 = addi %19, %c1_i32 : i32
    br ^bb5(%28 : i32)
  ^bb7:  // pred: ^bb5
    %29 = addi %15, %c1_i32 : i32
    br ^bb8(%29 : i32)
  ^bb8(%30: i32):  // 2 preds: ^bb7, ^bb9
    %31 = cmpi "slt", %30, %c2000_i32 : i32
    %32 = index_cast %30 : i32 to index
    cond_br %31, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %33 = sitofp %c0_i32 : i32 to f64
    store %33, %0[%17, %32] : memref<2000x2000xf64>
    %34 = addi %30, %c1_i32 : i32
    br ^bb8(%34 : i32)
  ^bb10:  // pred: ^bb8
    %35 = sitofp %c1_i32 : i32 to f64
    store %35, %0[%17, %17] : memref<2000x2000xf64>
    br ^bb3(%29 : i32)
  ^bb11(%36: i32):  // 2 preds: ^bb4, ^bb14
    %37 = cmpi "slt", %36, %c2000_i32 : i32
    %38 = index_cast %36 : i32 to index
    cond_br %37, ^bb12(%c0_i32 : i32), ^bb15(%c0_i32 : i32)
  ^bb12(%39: i32):  // 2 preds: ^bb11, ^bb13
    %40 = cmpi "slt", %39, %c2000_i32 : i32
    %41 = index_cast %39 : i32 to index
    cond_br %40, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %42 = sitofp %c0_i32 : i32 to f64
    store %42, %18[%38, %41] : memref<2000x2000xf64>
    %43 = addi %39, %c1_i32 : i32
    br ^bb12(%43 : i32)
  ^bb14:  // pred: ^bb12
    %44 = addi %36, %c1_i32 : i32
    br ^bb11(%44 : i32)
  ^bb15(%45: i32):  // 2 preds: ^bb11, ^bb17
    %46 = cmpi "slt", %45, %c2000_i32 : i32
    %47 = index_cast %45 : i32 to index
    cond_br %46, ^bb16(%c0_i32 : i32), ^bb21(%c0_i32 : i32)
  ^bb16(%48: i32):  // 2 preds: ^bb15, ^bb20
    %49 = cmpi "slt", %48, %c2000_i32 : i32
    %50 = index_cast %48 : i32 to index
    cond_br %49, ^bb18(%c0_i32 : i32), ^bb17
  ^bb17:  // pred: ^bb16
    %51 = addi %45, %c1_i32 : i32
    br ^bb15(%51 : i32)
  ^bb18(%52: i32):  // 2 preds: ^bb16, ^bb19
    %53 = cmpi "slt", %52, %c2000_i32 : i32
    %54 = index_cast %52 : i32 to index
    cond_br %53, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %55 = load %0[%50, %47] : memref<2000x2000xf64>
    %56 = load %0[%54, %47] : memref<2000x2000xf64>
    %57 = mulf %55, %56 : f64
    %58 = load %18[%50, %54] : memref<2000x2000xf64>
    %59 = addf %58, %57 : f64
    store %59, %18[%50, %54] : memref<2000x2000xf64>
    %60 = addi %52, %c1_i32 : i32
    br ^bb18(%60 : i32)
  ^bb20:  // pred: ^bb18
    %61 = addi %48, %c1_i32 : i32
    br ^bb16(%61 : i32)
  ^bb21(%62: i32):  // 2 preds: ^bb15, ^bb25
    %63 = cmpi "slt", %62, %c2000_i32 : i32
    %64 = index_cast %62 : i32 to index
    cond_br %63, ^bb23(%c0_i32 : i32), ^bb22
  ^bb22:  // pred: ^bb21
    %65 = get_global_memref @polybench_t_start : memref<1xf64>
    %66 = call @rtclock() : () -> f64
    store %66, %65[%c0] : memref<1xf64>
    %67 = alloca() : memref<1xf64>
    affine.for %arg2 = 0 to 2000 {
      call @S3(%67, %0, %c0, %arg2) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S5(%0, %c0, %arg2, %67) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
    }
    %68 = affine.load %0[%c1, %c0] : memref<2000x2000xf64>
    affine.store %68, %67[0] : memref<1xf64>
    %69 = affine.load %67[0] : memref<1xf64>
    %70 = affine.load %0[%c0, %c0] : memref<2000x2000xf64>
    %71 = divf %69, %70 : f64
    affine.store %71, %0[%c1, %c0] : memref<2000x2000xf64>
    affine.for %arg2 = 1 to 2000 {
      call @S3(%67, %0, %c1, %arg2) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S4(%67, %0, %c1, %arg2, %c0) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
      call @S5(%0, %c1, %arg2, %67) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
    }
    affine.for %arg2 = 2 to 2000 {
      call @S0(%67, %0, %arg2, %c0) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S2(%0, %arg2, %c0, %67) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      affine.for %arg3 = 1 to #map0(%arg2) {
        call @S0(%67, %0, %arg2, %arg3) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg4 = 0 to #map0(%arg3) {
          call @S1(%67, %0, %arg2, %arg3, %arg4) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
        }
        call @S2(%0, %arg2, %arg3, %67) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
      affine.for %arg3 = #map0(%arg2) to 2000 {
        call @S3(%67, %0, %arg2, %arg3) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg4 = 0 to #map0(%arg2) {
          call @S4(%67, %0, %arg2, %arg3, %arg4) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
        }
        call @S5(%0, %arg2, %arg3, %67) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
    }
    %72 = affine.load %67[0] : memref<1xf64>
    %73 = affine.load %0[-%c0 + symbol(%c2000) - 1, -%c0 + symbol(%c2000) - 1] : memref<2000x2000xf64>
    %74 = divf %72, %73 : f64
    affine.store %74, %2[-%c0 + symbol(%c2000) - 1] : memref<2000xf64>
    %75 = affine.load %3[-%c0 + symbol(%c2000) - 1] : memref<2000xf64>
    affine.store %75, %67[0] : memref<1xf64>
    affine.for %arg2 = 1 to 2000 {
      affine.for %arg3 = #map1(%arg2) to 2000 {
        call @S10(%67, %2, %arg2, %0, %arg3, %c2000) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, index) -> ()
      }
      call @S11(%2, %arg2, %c2000, %0, %67) : (memref<2000xf64>, index, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      call @S9(%67, %3, %arg2, %c2000) : (memref<1xf64>, memref<2000xf64>, index, index) -> ()
    }
    %76 = affine.load %1[%c0] : memref<2000xf64>
    affine.store %76, %67[0] : memref<1xf64>
    %77 = affine.load %67[0] : memref<1xf64>
    affine.store %77, %3[%c0] : memref<2000xf64>
    affine.for %arg2 = 1 to 2000 {
      call @S6(%67, %1, %arg2) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg3 = 0 to #map0(%arg2) {
        call @S7(%67, %3, %arg2, %0, %arg3) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index) -> ()
      }
      call @S8(%3, %arg2, %67) : (memref<2000xf64>, index, memref<1xf64>) -> ()
    }
    %78 = get_global_memref @polybench_t_end : memref<1xf64>
    %79 = call @rtclock() : () -> f64
    store %79, %78[%c0] : memref<1xf64>
    call @polybench_timer_print() : () -> ()
    %80 = cmpi "sgt", %arg0, %c42_i32 : i32
    %81 = scf.if %80 -> (i1) {
      %88 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
      %89 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %90 = llvm.mlir.constant(0 : index) : !llvm.i64
      %91 = llvm.getelementptr %89[%90, %90] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %92 = llvm.call @strcmp(%88, %91) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
      %93 = llvm.mlir.cast %92 : !llvm.i32 to i32
      %94 = trunci %93 : i32 to i1
      %95 = xor %94, %true : i1
      scf.yield %95 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %81 {
      call @print_array(%c2000_i32, %2) : (i32, memref<2000xf64>) -> ()
    }
    return %c0_i32 : i32
  ^bb23(%82: i32):  // 2 preds: ^bb21, ^bb24
    %83 = cmpi "slt", %82, %c2000_i32 : i32
    %84 = index_cast %82 : i32 to index
    cond_br %83, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %85 = load %18[%64, %84] : memref<2000x2000xf64>
    store %85, %0[%64, %84] : memref<2000x2000xf64>
    %86 = addi %82, %c1_i32 : i32
    br ^bb23(%86 : i32)
  ^bb25:  // pred: ^bb23
    %87 = addi %62, %c1_i32 : i32
    br ^bb21(%87 : i32)
  }
  func @init_array(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
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
    store %4, %arg3[%3] : memref<2000xf64>
    store %4, %arg4[%3] : memref<2000xf64>
    %5 = addi %1, %c1_i32 : i32
    %6 = sitofp %5 : i32 to f64
    %7 = divf %6, %0 : f64
    %8 = divf %7, %cst : f64
    %9 = sitofp %c4_i32 : i32 to f64
    %10 = addf %8, %9 : f64
    store %10, %arg2[%3] : memref<2000xf64>
    br ^bb1(%5 : i32)
  ^bb3(%11: i32):  // 2 preds: ^bb1, ^bb10
    %12 = cmpi "slt", %11, %arg0 : i32
    %13 = index_cast %11 : i32 to index
    cond_br %12, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %14 = alloc() : memref<2000x2000xf64>
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
    store %23, %arg1[%13, %17] : memref<2000x2000xf64>
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
    store %29, %arg1[%13, %28] : memref<2000x2000xf64>
    %30 = addi %26, %c1_i32 : i32
    br ^bb8(%30 : i32)
  ^bb10:  // pred: ^bb8
    %31 = sitofp %c1_i32 : i32 to f64
    store %31, %arg1[%13, %13] : memref<2000x2000xf64>
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
    store %38, %14[%34, %37] : memref<2000x2000xf64>
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
    %51 = load %arg1[%46, %43] : memref<2000x2000xf64>
    %52 = load %arg1[%50, %43] : memref<2000x2000xf64>
    %53 = mulf %51, %52 : f64
    %54 = load %14[%46, %50] : memref<2000x2000xf64>
    %55 = addf %54, %53 : f64
    store %55, %14[%46, %50] : memref<2000x2000xf64>
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
    %64 = load %14[%60, %63] : memref<2000x2000xf64>
    store %64, %arg1[%60, %63] : memref<2000x2000xf64>
    %65 = addi %61, %c1_i32 : i32
    br ^bb23(%65 : i32)
  ^bb25:  // pred: ^bb23
    %66 = addi %58, %c1_i32 : i32
    br ^bb21(%66 : i32)
  }
  func @polybench_timer_start() {
    %c0 = constant 0 : index
    %0 = get_global_memref @polybench_t_start : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = alloca() : memref<1xf64>
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to #map0(%arg5) {
        call @S0(%1, %arg1, %arg5, %arg6) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg7 = 0 to #map0(%arg6) {
          call @S1(%1, %arg1, %arg7, %arg6, %arg5) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
        }
        call @S2(%arg1, %arg5, %arg6, %1) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
      affine.for %arg6 = #map0(%arg5) to %0 {
        call @S3(%1, %arg1, %arg5, %arg6) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg7 = 0 to #map0(%arg5) {
          call @S4(%1, %arg1, %arg7, %arg6, %arg5) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
        }
        call @S5(%arg1, %arg5, %arg6, %1) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
    }
    affine.for %arg5 = 0 to %0 {
      call @S6(%1, %arg2, %arg5) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg6 = 0 to #map0(%arg5) {
        call @S7(%1, %arg4, %arg6, %arg1, %arg5) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index) -> ()
      }
      call @S8(%arg4, %arg5, %1) : (memref<2000xf64>, index, memref<1xf64>) -> ()
    }
    affine.for %arg5 = 0 to %0 {
      call @S9(%1, %arg4, %arg5, %0) : (memref<1xf64>, memref<2000xf64>, index, index) -> ()
      affine.for %arg6 = #map2(%arg5)[%0] to %0 {
        call @S10(%1, %arg3, %arg6, %arg1, %arg5, %0) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, index) -> ()
      }
      call @S11(%arg3, %arg5, %0, %arg1, %1) : (memref<2000xf64>, index, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
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
  func private @S0(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func private @S1(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg1[%arg4, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func private @S2(%arg0: memref<2000x2000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf64>
    %1 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    return
  }
  func private @S3(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func private @S4(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg1[%arg4, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func private @S5(%arg0: memref<2000x2000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf64>
    affine.store %0, %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    return
  }
  func private @S6(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2] : memref<2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func private @S7(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: memref<2000x2000xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg3[%arg4, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func private @S8(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    affine.store %0, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func private @S9(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[-%arg2 + symbol(%arg3) - 1] : memref<2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func private @S10(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: memref<2000x2000xf64>, %arg4: index, %arg5: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg3[-%arg4 + symbol(%arg5) - 1, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[0] : memref<1xf64>
    return
  }
  func private @S11(%arg0: memref<2000xf64>, %arg1: index, %arg2: index, %arg3: memref<2000x2000xf64>, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[0] : memref<1xf64>
    %1 = affine.load %arg3[-%arg1 + symbol(%arg2) - 1, -%arg1 + symbol(%arg2) - 1] : memref<2000x2000xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[-%arg1 + symbol(%arg2) - 1] : memref<2000xf64>
    return
  }
  func @kernel_ludcmp_new(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg5 = 0 to %1 {
      call @S3(%0, %arg1, %c0, %arg5) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S5(%arg1, %c0, %arg5, %0) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
    }
    %2 = affine.load %arg1[%c1, %c0] : memref<2000x2000xf64>
    affine.store %2, %0[0] : memref<1xf64>
    %3 = affine.load %0[0] : memref<1xf64>
    %4 = affine.load %arg1[%c0, %c0] : memref<2000x2000xf64>
    %5 = divf %3, %4 : f64
    affine.store %5, %arg1[%c1, %c0] : memref<2000x2000xf64>
    affine.for %arg5 = 1 to %1 {
      call @S3(%0, %arg1, %c1, %arg5) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S4(%0, %arg1, %c1, %arg5, %c0) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
      call @S5(%arg1, %c1, %arg5, %0) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
    }
    affine.for %arg5 = 2 to %1 {
      call @S0(%0, %arg1, %arg5, %c0) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S2(%arg1, %arg5, %c0, %0) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      affine.for %arg6 = 1 to #map0(%arg5) {
        call @S0(%0, %arg1, %arg5, %arg6) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg7 = 0 to #map0(%arg6) {
          call @S1(%0, %arg1, %arg5, %arg6, %arg7) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
        }
        call @S2(%arg1, %arg5, %arg6, %0) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
      affine.for %arg6 = #map0(%arg5) to %1 {
        call @S3(%0, %arg1, %arg5, %arg6) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg7 = 0 to #map0(%arg5) {
          call @S4(%0, %arg1, %arg5, %arg6, %arg7) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index) -> ()
        }
        call @S5(%arg1, %arg5, %arg6, %0) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
    }
    %6 = affine.load %0[0] : memref<1xf64>
    %7 = affine.load %arg1[-%c0 + symbol(%1) - 1, -%c0 + symbol(%1) - 1] : memref<2000x2000xf64>
    %8 = divf %6, %7 : f64
    affine.store %8, %arg3[-%c0 + symbol(%1) - 1] : memref<2000xf64>
    %9 = affine.load %arg4[-%c0 + symbol(%1) - 1] : memref<2000xf64>
    affine.store %9, %0[0] : memref<1xf64>
    affine.for %arg5 = 1 to %1 {
      affine.for %arg6 = #map2(%arg5)[%1] to %1 {
        call @S10(%0, %arg3, %arg5, %arg1, %arg6, %1) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, index) -> ()
      }
      call @S11(%arg3, %arg5, %1, %arg1, %0) : (memref<2000xf64>, index, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      call @S9(%0, %arg4, %arg5, %1) : (memref<1xf64>, memref<2000xf64>, index, index) -> ()
    }
    %10 = affine.load %arg2[%c0] : memref<2000xf64>
    affine.store %10, %0[0] : memref<1xf64>
    %11 = affine.load %0[0] : memref<1xf64>
    affine.store %11, %arg4[%c0] : memref<2000xf64>
    affine.for %arg5 = 1 to %1 {
      call @S6(%0, %arg2, %arg5) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg6 = 0 to #map0(%arg5) {
        call @S7(%0, %arg4, %arg5, %arg1, %arg6) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index) -> ()
      }
      call @S8(%arg4, %arg5, %0) : (memref<2000xf64>, index, memref<1xf64>) -> ()
    }
    return
  }
}

