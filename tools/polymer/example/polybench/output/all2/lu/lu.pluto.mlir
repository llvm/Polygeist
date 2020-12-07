#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> ((s0 - 2) floordiv 16 + 1)>
#map2 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map3 = affine_map<(d0) -> ((d0 - 1) floordiv 2 + 1)>
#map4 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32)>
#map5 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map6 = affine_map<(d0) -> (d0 * 32)>
#map7 = affine_map<(d0) -> (d0 * 32 + 32)>
#map8 = affine_map<(d0) -> (d0 * 32 + 1)>
#map9 = affine_map<(d0) -> (d0 floordiv 2)>
#map10 = affine_map<(d0) -> (d0 * 16)>
#map11 = affine_map<(d0)[s0] -> (s0, d0 * 16 + 32)>
#map12 = affine_map<(d0) -> (d0 * 16 + 1)>
#map13 = affine_map<()[s0] -> (s0 - 1)>
#map14 = affine_map<()[s0] -> (s0 - 2)>
#map15 = affine_map<(d0) -> (d0 * 16 + 2)>
#map16 = affine_map<(d0) -> ((d0 + 1) ceildiv 2)>
#map17 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map18 = affine_map<(d0, d1) -> (d0 - d1 + 1)>
#map19 = affine_map<(d0, d1, d2) -> (d0 * 32 - d1 * 32, d2 * 32 + 1)>
#map20 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32 + 32)>
#map21 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map22 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1)>
#map23 = affine_map<()[s0] -> ((s0 - 33) floordiv 32 + 1)>
#set0 = affine_set<(d0) : (d0 mod 2 == 0)>
#set1 = affine_set<(d0)[s0] : (d0 * 16 - (s0 - 2) == 0)>
#set2 = affine_set<()[s0] : ((s0 + 30) mod 32 == 0)>
#set3 = affine_set<(d0)[s0] : (-d0 + (s0 - 3) floordiv 16 >= 0)>
#set4 = affine_set<()[s0] : ((s0 + 15) mod 16 == 0)>
#set5 = affine_set<()[s0] : ((s0 + 31) mod 32 == 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2lf \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("A\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c40_i32 = constant 40 : i32
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<40x40xf64>
    call @init_array(%c40_i32, %0) : (i32, memref<40x40xf64>) -> ()
    call @kernel_lu_new(%c40_i32, %0) : (i32, memref<40x40xf64>) -> ()
    call @print_array(%c40_i32, %0) : (i32, memref<40x40xf64>) -> ()
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: memref<40x40xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb8
    %1 = cmpi "slt", %0, %arg0 : i32
    %2 = index_cast %0 : i32 to index
    cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %3 = alloc() : memref<40x40xf64>
    br ^bb9(%c0_i32 : i32)
  ^bb3(%4: i32):  // 2 preds: ^bb1, ^bb4
    %5 = cmpi "sle", %4, %0 : i32
    %6 = index_cast %4 : i32 to index
    cond_br %5, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %7 = subi %c0_i32, %4 : i32
    %8 = remi_signed %7, %arg0 : i32
    %9 = sitofp %8 : i32 to f64
    %10 = sitofp %arg0 : i32 to f64
    %11 = divf %9, %10 : f64
    %12 = sitofp %c1_i32 : i32 to f64
    %13 = addf %11, %12 : f64
    store %13, %arg1[%2, %6] : memref<40x40xf64>
    %14 = addi %4, %c1_i32 : i32
    br ^bb3(%14 : i32)
  ^bb5:  // pred: ^bb3
    %15 = addi %0, %c1_i32 : i32
    br ^bb6(%15 : i32)
  ^bb6(%16: i32):  // 2 preds: ^bb5, ^bb7
    %17 = cmpi "slt", %16, %arg0 : i32
    %18 = index_cast %16 : i32 to index
    cond_br %17, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %19 = sitofp %c0_i32 : i32 to f64
    store %19, %arg1[%2, %18] : memref<40x40xf64>
    %20 = addi %16, %c1_i32 : i32
    br ^bb6(%20 : i32)
  ^bb8:  // pred: ^bb6
    %21 = sitofp %c1_i32 : i32 to f64
    store %21, %arg1[%2, %2] : memref<40x40xf64>
    br ^bb1(%15 : i32)
  ^bb9(%22: i32):  // 2 preds: ^bb2, ^bb12
    %23 = cmpi "slt", %22, %arg0 : i32
    %24 = index_cast %22 : i32 to index
    cond_br %23, ^bb10(%c0_i32 : i32), ^bb13(%c0_i32 : i32)
  ^bb10(%25: i32):  // 2 preds: ^bb9, ^bb11
    %26 = cmpi "slt", %25, %arg0 : i32
    %27 = index_cast %25 : i32 to index
    cond_br %26, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %28 = sitofp %c0_i32 : i32 to f64
    store %28, %3[%24, %27] : memref<40x40xf64>
    %29 = addi %25, %c1_i32 : i32
    br ^bb10(%29 : i32)
  ^bb12:  // pred: ^bb10
    %30 = addi %22, %c1_i32 : i32
    br ^bb9(%30 : i32)
  ^bb13(%31: i32):  // 2 preds: ^bb9, ^bb15
    %32 = cmpi "slt", %31, %arg0 : i32
    %33 = index_cast %31 : i32 to index
    cond_br %32, ^bb14(%c0_i32 : i32), ^bb19(%c0_i32 : i32)
  ^bb14(%34: i32):  // 2 preds: ^bb13, ^bb18
    %35 = cmpi "slt", %34, %arg0 : i32
    %36 = index_cast %34 : i32 to index
    cond_br %35, ^bb16(%c0_i32 : i32), ^bb15
  ^bb15:  // pred: ^bb14
    %37 = addi %31, %c1_i32 : i32
    br ^bb13(%37 : i32)
  ^bb16(%38: i32):  // 2 preds: ^bb14, ^bb17
    %39 = cmpi "slt", %38, %arg0 : i32
    %40 = index_cast %38 : i32 to index
    cond_br %39, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %41 = load %arg1[%36, %33] : memref<40x40xf64>
    %42 = load %arg1[%40, %33] : memref<40x40xf64>
    %43 = mulf %41, %42 : f64
    %44 = load %3[%36, %40] : memref<40x40xf64>
    %45 = addf %44, %43 : f64
    store %45, %3[%36, %40] : memref<40x40xf64>
    %46 = addi %38, %c1_i32 : i32
    br ^bb16(%46 : i32)
  ^bb18:  // pred: ^bb16
    %47 = addi %34, %c1_i32 : i32
    br ^bb14(%47 : i32)
  ^bb19(%48: i32):  // 2 preds: ^bb13, ^bb23
    %49 = cmpi "slt", %48, %arg0 : i32
    %50 = index_cast %48 : i32 to index
    cond_br %49, ^bb21(%c0_i32 : i32), ^bb20
  ^bb20:  // pred: ^bb19
    return
  ^bb21(%51: i32):  // 2 preds: ^bb19, ^bb22
    %52 = cmpi "slt", %51, %arg0 : i32
    %53 = index_cast %51 : i32 to index
    cond_br %52, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %54 = load %3[%50, %53] : memref<40x40xf64>
    store %54, %arg1[%50, %53] : memref<40x40xf64>
    %55 = addi %51, %c1_i32 : i32
    br ^bb21(%55 : i32)
  ^bb23:  // pred: ^bb21
    %56 = addi %48, %c1_i32 : i32
    br ^bb19(%56 : i32)
  }
  func private @kernel_lu(%arg0: i32, %arg1: memref<40x40xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map0(%arg2) {
        affine.for %arg4 = 0 to #map0(%arg3) {
          call @S0(%arg1, %arg2, %arg3, %arg4) : (memref<40x40xf64>, index, index, index) -> ()
        }
        call @S1(%arg1, %arg2, %arg3) : (memref<40x40xf64>, index, index) -> ()
      }
      affine.for %arg3 = #map0(%arg2) to %0 {
        affine.for %arg4 = 0 to #map0(%arg2) {
          call @S2(%arg1, %arg2, %arg3, %arg4) : (memref<40x40xf64>, index, index, index) -> ()
        }
      }
    }
    return
  }
  func private @print_array(%arg0: i32, %arg1: memref<40x40xf64>) {
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
    %29 = cmpi "slt", %28, %arg0 : i32
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
    %39 = load %arg1[%15, %30] : memref<40x40xf64>
    %40 = llvm.mlir.cast %39 : f64 to !llvm.double
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %42 = addi %28, %c1_i32 : i32
    br ^bb3(%42 : i32)
  ^bb5:  // pred: ^bb3
    %43 = addi %13, %c1_i32 : i32
    br ^bb1(%43 : i32)
  }
  func private @S0(%arg0: memref<40x40xf64>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<40x40xf64>
    %1 = affine.load %arg0[%arg1, %arg3] : memref<40x40xf64>
    %2 = affine.load %arg0[%arg3, %arg2] : memref<40x40xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<40x40xf64>
    return
  }
  func private @S1(%arg0: memref<40x40xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<40x40xf64>
    %1 = affine.load %arg0[%arg2, %arg2] : memref<40x40xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[%arg1, %arg2] : memref<40x40xf64>
    return
  }
  func private @S2(%arg0: memref<40x40xf64>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<40x40xf64>
    %1 = affine.load %arg0[%arg1, %arg3] : memref<40x40xf64>
    %2 = affine.load %arg0[%arg3, %arg2] : memref<40x40xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<40x40xf64>
    return
  }
  func private @kernel_lu_new(%arg0: i32, %arg1: memref<40x40xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to #map1()[%0] {
      affine.for %arg3 = max #map2(%arg2)[%0] to #map3(%arg2) {
        affine.for %arg4 = 0 to #map0(%arg3) {
          affine.for %arg5 = #map4(%arg2, %arg3) to min #map5(%arg2, %arg3)[%0] {
            affine.for %arg6 = #map6(%arg3) to #map7(%arg3) {
              affine.for %arg7 = #map6(%arg4) to #map7(%arg4) {
                call @S0(%arg1, %arg5, %arg6, %arg7) : (memref<40x40xf64>, index, index, index) -> ()
              }
            }
          }
        }
        affine.for %arg4 = #map4(%arg2, %arg3) to min #map5(%arg2, %arg3)[%0] {
          %1 = affine.apply #map6(%arg3)
          call @S1(%arg1, %arg4, %1) : (memref<40x40xf64>, index, index) -> ()
          affine.for %arg5 = #map8(%arg3) to #map7(%arg3) {
            affine.for %arg6 = #map6(%arg3) to #map0(%arg5) {
              call @S0(%arg1, %arg4, %arg5, %arg6) : (memref<40x40xf64>, index, index, index) -> ()
            }
            call @S1(%arg1, %arg4, %arg5) : (memref<40x40xf64>, index, index) -> ()
          }
        }
      }
      affine.if #set0(%arg2) {
        affine.for %arg3 = 0 to #map9(%arg2) {
          affine.for %arg4 = #map10(%arg2) to min #map11(%arg2)[%0] {
            affine.for %arg5 = #map6(%arg3) to #map7(%arg3) {
              %1 = affine.apply #map10(%arg2)
              call @S2(%arg1, %1, %arg4, %arg5) : (memref<40x40xf64>, index, index, index) -> ()
            }
          }
          affine.for %arg4 = #map12(%arg2) to min #map11(%arg2)[%0] {
            affine.for %arg5 = #map10(%arg2) to #map0(%arg4) {
              affine.for %arg6 = #map6(%arg3) to #map7(%arg3) {
                call @S0(%arg1, %arg4, %arg5, %arg6) : (memref<40x40xf64>, index, index, index) -> ()
              }
            }
            affine.for %arg5 = #map0(%arg4) to min #map11(%arg2)[%0] {
              affine.for %arg6 = #map6(%arg3) to #map7(%arg3) {
                call @S2(%arg1, %arg4, %arg5, %arg6) : (memref<40x40xf64>, index, index, index) -> ()
              }
            }
          }
        }
        affine.if #set1(%arg2)[%0] {
          affine.if #set2()[%0] {
            %1 = affine.apply #map13()[%0]
            %2 = affine.apply #map14()[%0]
            call @S1(%arg1, %1, %2) : (memref<40x40xf64>, index, index) -> ()
            %3 = affine.apply #map13()[%0]
            %4 = affine.apply #map13()[%0]
            %5 = affine.apply #map14()[%0]
            call @S2(%arg1, %3, %4, %5) : (memref<40x40xf64>, index, index, index) -> ()
          }
        }
        affine.if #set3(%arg2)[%0] {
          %1 = affine.apply #map12(%arg2)
          %2 = affine.apply #map10(%arg2)
          call @S1(%arg1, %1, %2) : (memref<40x40xf64>, index, index) -> ()
          affine.for %arg3 = #map12(%arg2) to min #map11(%arg2)[%0] {
            %3 = affine.apply #map12(%arg2)
            %4 = affine.apply #map10(%arg2)
            call @S2(%arg1, %3, %arg3, %4) : (memref<40x40xf64>, index, index, index) -> ()
          }
          affine.for %arg3 = #map15(%arg2) to min #map11(%arg2)[%0] {
            %3 = affine.apply #map10(%arg2)
            call @S1(%arg1, %arg3, %3) : (memref<40x40xf64>, index, index) -> ()
            affine.for %arg4 = #map12(%arg2) to #map0(%arg3) {
              affine.for %arg5 = #map10(%arg2) to #map0(%arg4) {
                call @S0(%arg1, %arg3, %arg4, %arg5) : (memref<40x40xf64>, index, index, index) -> ()
              }
              call @S1(%arg1, %arg3, %arg4) : (memref<40x40xf64>, index, index) -> ()
            }
            affine.for %arg4 = #map0(%arg3) to min #map11(%arg2)[%0] {
              affine.for %arg5 = #map10(%arg2) to #map0(%arg3) {
                call @S2(%arg1, %arg3, %arg4, %arg5) : (memref<40x40xf64>, index, index, index) -> ()
              }
            }
          }
        }
      }
      affine.for %arg3 = #map16(%arg2) to min #map17(%arg2)[%0] {
        affine.for %arg4 = 0 to #map18(%arg2, %arg3) {
          affine.for %arg5 = max #map19(%arg2, %arg3, %arg4) to #map20(%arg2, %arg3) {
            affine.for %arg6 = #map6(%arg3) to min #map21(%arg3)[%0] {
              affine.for %arg7 = #map6(%arg4) to min #map22(%arg4, %arg5) {
                call @S2(%arg1, %arg5, %arg6, %arg7) : (memref<40x40xf64>, index, index, index) -> ()
              }
            }
          }
        }
      }
    }
    affine.if #set4()[%0] {
      affine.if #set5()[%0] {
        affine.for %arg2 = 0 to #map23()[%0] {
          affine.for %arg3 = #map6(%arg2) to #map7(%arg2) {
            %1 = affine.apply #map13()[%0]
            %2 = affine.apply #map13()[%0]
            call @S2(%arg1, %1, %2, %arg3) : (memref<40x40xf64>, index, index, index) -> ()
          }
        }
      }
    }
    return
  }
}

