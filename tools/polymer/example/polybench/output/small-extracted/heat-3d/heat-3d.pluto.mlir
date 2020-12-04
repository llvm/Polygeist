#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 ceildiv 2)>
#map2 = affine_map<(d0)[s0] -> ((s0 + 38) floordiv 32 + 1, (d0 * 16 + s0 + 13) floordiv 32 + 1, (d0 * 32 + s0 + 26) floordiv 32 + 1)>
#map3 = affine_map<()[s0] -> ((s0 + 28) floordiv 32 + 1)>
#map4 = affine_map<(d0) -> (31, d0 * 32)>
#map5 = affine_map<(d0)[s0] -> (s0 + 29, d0 * 32 + 32)>
#map6 = affine_map<()[s0] -> (s0 - 2)>
#map7 = affine_map<(d0, d1)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32)>
#map8 = affine_map<(d0, d1)[s0] -> ((s0 + 38) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1)>
#map9 = affine_map<(d0, d1) -> (d0 * 32, d1 * -32 + d0 * 32 + 29)>
#map10 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 61)>
#map11 = affine_map<(d0) -> (d0 - 30)>
#map12 = affine_map<(d0) -> (d0 * 32)>
#map13 = affine_map<(d0, d1, d2)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32, (d2 * 32 - s0 - 27) ceildiv 32)>
#map14 = affine_map<(d0, d1, d2)[s0] -> ((s0 + 38) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1, (d2 * 32 + s0 + 27) floordiv 32 + 1)>
#map15 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 - s0 + 3)>
#map16 = affine_map<(d0, d1) -> (d0 * 32 + 1, d1 * 32 + 32)>
#map17 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 1) ceildiv 2)>
#map18 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 + s0 - 2)>
#map19 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - s0 + 3, d2 * -32 + d0 * 32 + d1 * 64 - s0 * 2 - 27)>
#map20 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 + 5)>
#map21 = affine_map<(d0) -> (d0 * 32 + 32)>
#map22 = affine_map<(d0) -> (34, d0 * 32)>
#map23 = affine_map<(d0)[s0] -> (s0 + 32, d0 * 32 + 32)>
#map24 = affine_map<(d0) -> (d0 - 33)>
#map25 = affine_map<(d0, d1, d2, d3)[s0] -> (1, (d0 * 32 - s0 + 2) ceildiv 2, (d1 * 32 - s0 + 2) ceildiv 2, (d2 * 32 - s0 + 2) ceildiv 2, d3 * 8, d3 * 16 - d0 * 16 + 1)>
#map26 = affine_map<(d0, d1, d2, d3)[s0] -> (21, d0 * 16 - d1 * 16 + s0 floordiv 2 + 15, d0 * 8 + 16, d1 * 16 + 15, d2 * 16 + 15, d3 * 16 + 15)>
#map27 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 1)>
#map28 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0 - 1)>
#map29 = affine_map<(d0, d1) -> (d0 * -2 + d1)>
#map30 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 31)>
#map31 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 - 29)>
#map32 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 29)>
#map33 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 4 + 1, d2 * 2 + s0 - 1)>
#map34 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
#map35 = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
#map36 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0)>
#map37 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 + 1)>
#map38 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 2 + s0, d2 * -32 + d0 * 32 + d1 * 4 + 3)>
#map39 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - d2 * 32 + s0 + 30)>
#map40 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 * 32 + s0 * 2 + 28)>
#map41 = affine_map<(d0, d1)[s0] -> ((d0 * 32 - d1 * 32 + s0 + 29) ceildiv 2)>
#map42 = affine_map<(d0, d1, d2)[s0] -> (d0 * -32 + d1 * 32 + d2 - s0 - 29)>
#map43 = affine_map<(d0, d1) -> (31, d0 * 32, d1 * -32 + d0 * 32 + 29)>
#map44 = affine_map<(d0, d1)[s0] -> (s0 + 29, d0 * 32 + 32, d1 * -32 + d0 * 32 + 61)>
#map45 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 32)>
#map46 = affine_map<(d0) -> (d0 * 8 + 15)>
#map47 = affine_map<(d0, d1) -> (d0 * -16 + d1 * 32)>
#map48 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 31)>
#map49 = affine_map<(d0, d1) -> (42, d0 * 32, d1 * -32 + d0 * 32 + 51)>
#map50 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 83)>
#map51 = affine_map<(d0) -> (42, d0 * 32)>
#map52 = affine_map<(d0) -> (d0 - 41)>
#map53 = affine_map<(d0) -> ((d0 + 1) ceildiv 2)>
#map54 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 29) floordiv 32 + 1)>
#map55 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 30)>
#map56 = affine_map<()[s0] -> ((s0 + 39) floordiv 32 + 1)>
#map57 = affine_map<(d0)[s0] -> (s0 + 40, d0 * 32 + 32)>
#map58 = affine_map<()[s0] -> ((s0 - 5) floordiv 32 + 1)>
#map59 = affine_map<(d0) -> (3, d0 * 32)>
#map60 = affine_map<(d0)[s0] -> (s0 + 1, d0 * 32 + 32)>
#map61 = affine_map<(d0) -> (d0 - 2)>
#set0 = affine_set<(d0, d1)[s0] : (d0 - 1 == 0, d1 * 32 - (s0 + 28) == 0)>
#set1 = affine_set<()[s0] : ((s0 + 28) mod 32 == 0)>
#set2 = affine_set<(d0, d1, d2)[s0] : (-d0 + 1 >= 0, -d1 + (s0 - 4) floordiv 32 >= 0, d2 * 32 - (s0 + 28) == 0)>
#set3 = affine_set<(d0, d1)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set4 = affine_set<()[s0] : ((s0 + 1) mod 2 == 0)>
#set5 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set6 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - d3 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set7 = affine_set<(d0, d1, d2, d3)[s0] : (d0 - 2 == 0, d1 - 1 == 0, -d2 + (s0 + 31) floordiv 32 >= 0, -d3 + (s0 + 31) floordiv 32 >= 0)>
#set8 = affine_set<(d0, d1, d2) : (d0 - (d1 * 16 + d2 - 16) ceildiv 16 >= 0, d2 floordiv 16 - d1 >= 0)>
#set9 = affine_set<(d0, d1) : (d1 floordiv 16 - d0 >= 0)>
#set10 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set11 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 + d2 * 2 - s0 + 1) floordiv 32 - d0 >= 0, d1 - (d2 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set12 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d1 * 32 - s0 + 11) floordiv 32 - d0 >= 0, (d1 * 32 + d2 * 32 - s0 - 1) floordiv 32 - d0 >= 0, (d1 * 32 + d3 * 32 - s0 - 1) floordiv 32 - d0 >= 0)>
#set13 = affine_set<(d0, d1, d2, d3)[s0] : (d0 >= 0, d0 - (d1 * 32 - s0 + 1) ceildiv 32 >= 0, d2 == 0, d3 - 1 >= 0)>
#set14 = affine_set<(d0, d1, d2)[s0] : (d0 >= 0, d0 - (d1 * 32 - s0 + 1) ceildiv 32 >= 0, d2 == 0)>
#set15 = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 - 1 >= 0, d3 - 1 >= 0)>
#set16 = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 - 1 >= 0)>
#set17 = affine_set<(d0, d1) : (d0 >= 0, d1 == 0)>
#set18 = affine_set<(d0, d1, d2) : (-d0 >= 0, d1 * 2 - d0 - 1 >= 0, d2 * 2 - d0 - 1 >= 0)>
#set19 = affine_set<(d0)[s0] : ((d0 * 16 + s0 + 29) mod 32 == 0)>
#set20 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set21 = affine_set<()[s0] : ((s0 + 7) mod 32 == 0)>
#set22 = affine_set<(d0, d1) : (d0 == 0, d1 == 0)>
#set23 = affine_set<(d0, d1) : (-d0 >= 0, d1 * 2 - d0 - 1 >= 0)>
#set24 = affine_set<(d0) : (d0 - 1 >= 0)>
#set25 = affine_set<(d0) : (d0 + 1 == 0)>
#set26 = affine_set<()[s0] : ((s0 + 27) mod 32 == 0)>
#set27 = affine_set<(d0) : (d0 - 2 == 0)>
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
    %c10_i32 = constant 10 : i32
    %c20_i32 = constant 20 : i32
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<10x10x10xf64>
    %1 = alloc() : memref<10x10x10xf64>
    call @init_array(%c10_i32, %0, %1) : (i32, memref<10x10x10xf64>, memref<10x10x10xf64>) -> ()
    call @kernel_heat_3d_new(%c20_i32, %c10_i32, %0, %1) : (i32, i32, memref<10x10x10xf64>, memref<10x10x10xf64>) -> ()
    call @print_array(%c10_i32, %0) : (i32, memref<10x10x10xf64>) -> ()
    return %c0_i32 : i32
  }
  func @init_array(%arg0: i32, %arg1: memref<10x10x10xf64>, %arg2: memref<10x10x10xf64>) {
    %c0_i32 = constant 0 : i32
    %c10_i32 = constant 10 : i32
    %c1_i32 = constant 1 : i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb4
    %1 = cmpi "slt", %0, %arg0 : i32
    %2 = index_cast %0 : i32 to index
    cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    return
  ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb7
    %4 = cmpi "slt", %3, %arg0 : i32
    %5 = index_cast %3 : i32 to index
    cond_br %4, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %6 = addi %0, %c1_i32 : i32
    br ^bb1(%6 : i32)
  ^bb5(%7: i32):  // 2 preds: ^bb3, ^bb6
    %8 = cmpi "slt", %7, %arg0 : i32
    %9 = index_cast %7 : i32 to index
    cond_br %8, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %10 = addi %0, %3 : i32
    %11 = subi %arg0, %7 : i32
    %12 = addi %10, %11 : i32
    %13 = sitofp %12 : i32 to f64
    %14 = sitofp %c10_i32 : i32 to f64
    %15 = mulf %13, %14 : f64
    %16 = sitofp %arg0 : i32 to f64
    %17 = divf %15, %16 : f64
    store %17, %arg2[%2, %5, %9] : memref<10x10x10xf64>
    %18 = load %arg2[%2, %5, %9] : memref<10x10x10xf64>
    store %18, %arg1[%2, %5, %9] : memref<10x10x10xf64>
    %19 = addi %7, %c1_i32 : i32
    br ^bb5(%19 : i32)
  ^bb7:  // pred: ^bb5
    %20 = addi %3, %c1_i32 : i32
    br ^bb3(%20 : i32)
  }
  func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<10x10x10xf64>, %arg3: memref<10x10x10xf64>) {
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to 21 {
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          affine.for %arg7 = 1 to #map0()[%0] {
            call @S0(%arg3, %arg5, %arg6, %arg7, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
          }
        }
      }
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          affine.for %arg7 = 1 to #map0()[%0] {
            call @S1(%arg2, %arg5, %arg6, %arg7, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
          }
        }
      }
    }
    return
  }
  func @print_array(%arg0: i32, %arg1: memref<10x10x10xf64>) {
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
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb4
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
  ^bb3(%28: i32):  // 2 preds: ^bb1, ^bb7
    %29 = cmpi "slt", %28, %arg0 : i32
    %30 = index_cast %28 : i32 to index
    cond_br %29, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %31 = addi %13, %c1_i32 : i32
    br ^bb1(%31 : i32)
  ^bb5(%32: i32):  // 2 preds: ^bb3, ^bb6
    %33 = cmpi "slt", %32, %arg0 : i32
    %34 = index_cast %32 : i32 to index
    cond_br %33, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %35 = muli %13, %arg0 : i32
    %36 = muli %35, %arg0 : i32
    %37 = muli %28, %arg0 : i32
    %38 = addi %36, %37 : i32
    %39 = addi %38, %32 : i32
    %40 = remi_signed %39, %c20_i32 : i32
    %41 = cmpi "eq", %40, %c0_i32 : i32
    scf.if %41 {
      %51 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %52 = llvm.load %51 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %53 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
      %54 = llvm.getelementptr %53[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %55 = llvm.call @fprintf(%52, %54) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %42 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %43 = llvm.load %42 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %44 = llvm.mlir.addressof @str4 : !llvm.ptr<array<8 x i8>>
    %45 = llvm.getelementptr %44[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %46 = load %arg1[%15, %30, %34] : memref<10x10x10xf64>
    %47 = llvm.mlir.cast %46 : f64 to !llvm.double
    %48 = llvm.call @fprintf(%43, %45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %49 = addi %32, %c1_i32 : i32
    br ^bb5(%49 : i32)
  ^bb7:  // pred: ^bb5
    %50 = addi %28, %c1_i32 : i32
    br ^bb3(%50 : i32)
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<10x10x10xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<10x10x10xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.250000e-01 : f64
    %0 = affine.load %arg4[%arg1 + 1, %arg2, %arg3] : memref<10x10x10xf64>
    %1 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %2 = affine.load %arg4[%arg1 - 1, %arg2, %arg3] : memref<10x10x10xf64>
    %3 = affine.load %arg4[%arg1, %arg2 + 1, %arg3] : memref<10x10x10xf64>
    %4 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %5 = affine.load %arg4[%arg1, %arg2 - 1, %arg3] : memref<10x10x10xf64>
    %6 = affine.load %arg4[%arg1, %arg2, %arg3 + 1] : memref<10x10x10xf64>
    %7 = mulf %cst, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst_0, %9 : f64
    %11 = mulf %cst, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst_0, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %17 = mulf %cst, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[%arg1, %arg2, %arg3 - 1] : memref<10x10x10xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst_0, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    return
  }
  func private @S1(%arg0: memref<10x10x10xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<10x10x10xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.250000e-01 : f64
    %0 = affine.load %arg4[%arg1 + 1, %arg2, %arg3] : memref<10x10x10xf64>
    %1 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %2 = affine.load %arg4[%arg1 - 1, %arg2, %arg3] : memref<10x10x10xf64>
    %3 = affine.load %arg4[%arg1, %arg2 + 1, %arg3] : memref<10x10x10xf64>
    %4 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %5 = affine.load %arg4[%arg1, %arg2 - 1, %arg3] : memref<10x10x10xf64>
    %6 = affine.load %arg4[%arg1, %arg2, %arg3 + 1] : memref<10x10x10xf64>
    %7 = mulf %cst, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst_0, %9 : f64
    %11 = mulf %cst, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst_0, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %17 = mulf %cst, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[%arg1, %arg2, %arg3 - 1] : memref<10x10x10xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst_0, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[%arg1, %arg2, %arg3] : memref<10x10x10xf64>
    return
  }
  func @kernel_heat_3d_new(%arg0: i32, %arg1: i32, %arg2: memref<10x10x10xf64>, %arg3: memref<10x10x10xf64>) {
    %c16 = constant 16 : index
    %c15 = constant 15 : index
    %c1 = constant 1 : index
    %c20 = constant 20 : index
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = -1 to 3 {
      affine.for %arg5 = #map1(%arg4) to min #map2(%arg4)[%0] {
        affine.if #set0(%arg4, %arg5)[%0] {
          affine.for %arg6 = 0 to #map3()[%0] {
            affine.for %arg7 = max #map4(%arg6) to min #map5(%arg6)[%0] {
              affine.if #set1()[%0] {
                %1 = affine.apply #map6()[%0]
                call @S0(%arg3, %c15, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.for %arg6 = max #map7(%arg4, %arg5)[%0] to min #map8(%arg4, %arg5)[%0] {
          affine.if #set2(%arg4, %arg5, %arg6)[%0] {
            affine.for %arg7 = max #map9(%arg4, %arg5) to min #map10(%arg4, %arg5) {
              affine.if #set1()[%0] {
                %1 = affine.apply #map11(%arg7)
                %2 = affine.apply #map6()[%0]
                call @S0(%arg3, %c15, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set0(%arg4, %arg5)[%0] {
            affine.for %arg7 = #map12(%arg6) to min #map5(%arg6)[%0] {
              affine.if #set1()[%0] {
                %1 = affine.apply #map6()[%0]
                %2 = affine.apply #map11(%arg7)
                call @S0(%arg3, %c15, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.for %arg7 = max #map13(%arg4, %arg5, %arg6)[%0] to min #map14(%arg4, %arg5, %arg6)[%0] {
            affine.if #set3(%arg4, %arg5)[%0] {
              affine.if #set4()[%0] {
                affine.for %arg8 = max #map15(%arg5, %arg6)[%0] to min #map16(%arg5, %arg6) {
                  affine.for %arg9 = max #map15(%arg5, %arg7)[%0] to min #map16(%arg5, %arg7) {
                    %1 = affine.apply #map17(%arg5)[%0]
                    %2 = affine.apply #map6()[%0]
                    %3 = affine.apply #map18(%arg5, %arg8)[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set5(%arg4, %arg5, %arg6)[%0] {
              affine.if #set4()[%0] {
                affine.for %arg8 = max #map19(%arg4, %arg5, %arg6)[%0] to min #map20(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map15(%arg6, %arg7)[%0] to min #map16(%arg6, %arg7) {
                    %1 = affine.apply #map17(%arg6)[%0]
                    %2 = affine.apply #map18(%arg6, %arg8)[%0]
                    %3 = affine.apply #map6()[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set6(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set4()[%0] {
                affine.for %arg8 = max #map19(%arg4, %arg5, %arg7)[%0] to min #map20(%arg4, %arg5, %arg7)[%0] {
                  affine.for %arg9 = max #map15(%arg6, %arg7)[%0] to #map21(%arg6) {
                    %1 = affine.apply #map17(%arg7)[%0]
                    %2 = affine.apply #map18(%arg7, %arg8)[%0]
                    %3 = affine.apply #map18(%arg7, %arg9)[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set7(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map22(%arg6) to min #map23(%arg6)[%0] {
                affine.for %arg9 = max #map22(%arg7) to min #map23(%arg7)[%0] {
                  %1 = affine.apply #map24(%arg8)
                  call @S1(%arg2, %c16, %c1, %1, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.for %arg8 = max #map25(%arg4, %arg5, %arg6, %arg7)[%0] to min #map26(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set8(%arg4, %arg5, %arg8) {
                affine.for %arg9 = max #map27(%arg6, %arg8) to min #map28(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map27(%arg7, %arg8) to min #map28(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map29(%arg8, %arg9)
                    call @S0(%arg3, %arg8, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map30(%arg4, %arg5, %arg8) to #map31(%arg4, %arg5, %arg8) {
                affine.for %arg10 = max #map27(%arg6, %arg8) to min #map28(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map27(%arg7, %arg8) to min #map28(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map29(%arg8, %arg9)
                    %2 = affine.apply #map29(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map32(%arg4, %arg5, %arg8) to min #map33(%arg4, %arg5, %arg8)[%0] {
                affine.if #set9(%arg6, %arg8) {
                  affine.for %arg10 = max #map27(%arg7, %arg8) to min #map28(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map29(%arg8, %arg9)
                    call @S0(%arg3, %arg8, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
                affine.for %arg10 = max #map34(%arg6, %arg8) to min #map28(%arg6, %arg8)[%0] {
                  affine.if #set9(%arg7, %arg8) {
                    %1 = affine.apply #map29(%arg8, %arg9)
                    %2 = affine.apply #map29(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                  affine.for %arg11 = max #map34(%arg7, %arg8) to min #map28(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map35(%arg8, %arg9)
                    %2 = affine.apply #map35(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                    %3 = affine.apply #map29(%arg8, %arg9)
                    %4 = affine.apply #map29(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %3, %4, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                  affine.if #set10(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map35(%arg8, %arg9)
                    %2 = affine.apply #map35(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
                affine.if #set10(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map34(%arg7, %arg8) to min #map36(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map35(%arg8, %arg9)
                    %2 = affine.apply #map6()[%0]
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = #map37(%arg4, %arg5, %arg8) to min #map38(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg10 = max #map34(%arg6, %arg8) to min #map36(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map34(%arg7, %arg8) to min #map36(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map35(%arg8, %arg9)
                    %2 = affine.apply #map35(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.if #set11(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg9 = max #map34(%arg6, %arg8) to min #map36(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map34(%arg7, %arg8) to min #map36(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map6()[%0]
                    %2 = affine.apply #map35(%arg8, %arg9)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set12(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set4()[%0] {
                affine.for %arg8 = max #map39(%arg4, %arg5, %arg6)[%0] to min #map40(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map39(%arg4, %arg5, %arg7)[%0] to min #map40(%arg4, %arg5, %arg7)[%0] {
                    %1 = affine.apply #map41(%arg4, %arg5)[%0]
                    %2 = affine.apply #map6()[%0]
                    %3 = affine.apply #map42(%arg4, %arg5, %arg8)[%0]
                    call @S0(%arg3, %1, %2, %3, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set13(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map43(%arg4, %arg5) to min #map44(%arg4, %arg5)[%0] {
                affine.for %arg9 = #map12(%arg7) to min #map5(%arg7)[%0] {
                  %1 = affine.apply #map11(%arg8)
                  call @S0(%arg3, %c15, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.if #set14(%arg4, %arg5, %arg7)[%0] {
              affine.for %arg8 = max #map43(%arg4, %arg5) to min #map44(%arg4, %arg5)[%0] {
                affine.for %arg9 = max #map4(%arg6) to min #map5(%arg6)[%0] {
                  %1 = affine.apply #map11(%arg8)
                  %2 = affine.apply #map11(%arg9)
                  call @S0(%arg3, %c15, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.if #set15(%arg4, %arg5, %arg6, %arg7) {
              affine.for %arg8 = #map12(%arg6) to min #map5(%arg6)[%0] {
                affine.for %arg9 = #map12(%arg7) to min #map5(%arg7)[%0] {
                  %1 = affine.apply #map11(%arg8)
                  call @S0(%arg3, %c15, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
          affine.if #set16(%arg4, %arg5, %arg6) {
            affine.if #set1()[%0] {
              affine.for %arg7 = #map12(%arg6) to #map21(%arg6) {
                %1 = affine.apply #map11(%arg7)
                call @S0(%arg3, %c15, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set17(%arg4, %arg6) {
            affine.if #set1()[%0] {
              affine.for %arg7 = max #map43(%arg4, %arg5) to min #map10(%arg4, %arg5) {
                %1 = affine.apply #map11(%arg7)
                call @S0(%arg3, %c15, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set18(%arg4, %arg5, %arg6) {
            affine.if #set19(%arg4)[%0] {
              affine.for %arg7 = max #map45(%arg4, %arg6) to #map21(%arg6) {
                %1 = affine.apply #map46(%arg4)
                %2 = affine.apply #map47(%arg4, %arg5)
                %3 = affine.apply #map48(%arg4, %arg7)
                call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set20(%arg4, %arg6) {
            affine.if #set21()[%0] {
              affine.for %arg7 = max #map49(%arg4, %arg5) to min #map50(%arg4, %arg5) {
                affine.for %arg8 = max #map51(%arg6) to #map21(%arg6) {
                  %1 = affine.apply #map52(%arg7)
                  %2 = affine.apply #map52(%arg8)
                  call @S1(%arg2, %c20, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set22(%arg4, %arg5) {
          affine.if #set1()[%0] {
            affine.for %arg6 = 0 to #map3()[%0] {
              affine.for %arg7 = max #map4(%arg6) to min #map5(%arg6)[%0] {
                %1 = affine.apply #map6()[%0]
                call @S0(%arg3, %c15, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.if #set23(%arg4, %arg5) {
          affine.if #set19(%arg4)[%0] {
            affine.for %arg6 = #map53(%arg4) to #map54(%arg4)[%0] {
              affine.for %arg7 = max #map45(%arg4, %arg6) to min #map55(%arg4, %arg6)[%0] {
                %1 = affine.apply #map46(%arg4)
                %2 = affine.apply #map47(%arg4, %arg5)
                %3 = affine.apply #map6()[%0]
                call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.if #set24(%arg4) {
          affine.if #set21()[%0] {
            affine.for %arg6 = 1 to #map56()[%0] {
              affine.for %arg7 = max #map49(%arg4, %arg5) to min #map50(%arg4, %arg5) {
                affine.for %arg8 = max #map51(%arg6) to min #map57(%arg6)[%0] {
                  %1 = affine.apply #map52(%arg7)
                  %2 = affine.apply #map6()[%0]
                  call @S1(%arg2, %c20, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set25(%arg4) {
        affine.if #set26()[%0] {
          affine.for %arg5 = 0 to #map58()[%0] {
            affine.for %arg6 = 0 to #map58()[%0] {
              affine.for %arg7 = max #map59(%arg5) to min #map60(%arg5)[%0] {
                affine.for %arg8 = max #map59(%arg6) to min #map60(%arg6)[%0] {
                  %1 = affine.apply #map6()[%0]
                  %2 = affine.apply #map61(%arg7)
                  call @S0(%arg3, %c1, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set27(%arg4) {
        affine.if #set21()[%0] {
          affine.for %arg5 = 1 to #map56()[%0] {
            affine.for %arg6 = 1 to #map56()[%0] {
              affine.for %arg7 = max #map51(%arg5) to min #map57(%arg5)[%0] {
                affine.for %arg8 = max #map51(%arg6) to min #map57(%arg6)[%0] {
                  %1 = affine.apply #map6()[%0]
                  %2 = affine.apply #map52(%arg7)
                  call @S1(%arg2, %c20, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

