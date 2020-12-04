#map0 = affine_map<(d0) -> (d0 ceildiv 2)>
#map1 = affine_map<(d0) -> (2, (d0 * 16 + 23) floordiv 32 + 1, d0 + 2)>
#map2 = affine_map<(d0) -> (31, d0 * 32)>
#map3 = affine_map<(d0) -> (39, d0 * 32 + 32)>
#map4 = affine_map<(d0, d1) -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - 37) ceildiv 32)>
#map5 = affine_map<(d0, d1) -> (2, (d0 * 16 + 38) floordiv 32 + 1, d1 + 2)>
#map6 = affine_map<(d0, d1) -> (d0 * 32, d1 * -32 + d0 * 32 + 29)>
#map7 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 61)>
#map8 = affine_map<(d0) -> (d0 - 30)>
#map9 = affine_map<(d0) -> (d0 * 32)>
#map10 = affine_map<(d0, d1, d2) -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - 37) ceildiv 32, (d2 * 32 - 37) ceildiv 32)>
#map11 = affine_map<(d0, d1, d2) -> (2, (d0 * 16 + 38) floordiv 32 + 1, d1 + 2, d2 + 2)>
#map12 = affine_map<(d0, d1) -> (d0 * 32, d1 * 32 - 7)>
#map13 = affine_map<(d0, d1) -> (d0 * 32 + 1, d1 * 32 + 32)>
#map14 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 1) ceildiv 2)>
#map15 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 + s0 - 2)>
#map16 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 - 7, d2 * -32 + d0 * 32 + d1 * 64 - 47)>
#map17 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 - 15)>
#map18 = affine_map<(d0) -> (d0 * 32 + 32)>
#map19 = affine_map<(d0) -> (34, d0 * 32)>
#map20 = affine_map<(d0) -> (42, d0 * 32 + 32)>
#map21 = affine_map<(d0) -> (d0 - 33)>
#map22 = affine_map<(d0, d1, d2, d3) -> (1, (d0 * 32 - 8) ceildiv 2, (d1 * 32 - 8) ceildiv 2, (d2 * 32 - 8) ceildiv 2, d3 * 8, d3 * 16 - d0 * 16 + 1)>
#map23 = affine_map<(d0, d1, d2, d3) -> (21, d0 * 16 - d1 * 16 + 20, d0 * 8 + 16, d1 * 16 + 15, d2 * 16 + 15, d3 * 16 + 15)>
#map24 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 1)>
#map25 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * 2 + 9)>
#map26 = affine_map<(d0, d1) -> (d0 * -2 + d1)>
#map27 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 31)>
#map28 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 - 29)>
#map29 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 29)>
#map30 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 4 + 1, d2 * 2 + 9)>
#map31 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
#map32 = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
#map33 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * 2 + 10)>
#map34 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 + 1)>
#map35 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * 2 + 10, d2 * -32 + d0 * 32 + d1 * 4 + 3)>
#map36 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 - d2 * 32 + 40)>
#map37 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * 32 - d2 * 32 + 48)>
#map38 = affine_map<(d0, d1)[s0] -> ((d0 * 32 - d1 * 32 + s0 + 29) ceildiv 2)>
#map39 = affine_map<(d0, d1, d2)[s0] -> (d0 * -32 + d1 * 32 + d2 - s0 - 29)>
#map40 = affine_map<(d0, d1) -> (31, d0 * 32, d1 * -32 + d0 * 32 + 29)>
#map41 = affine_map<(d0, d1) -> (39, d0 * 32 + 32, d1 * -32 + d0 * 32 + 61)>
#map42 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 32)>
#map43 = affine_map<(d0) -> (d0 * 8 + 15)>
#map44 = affine_map<(d0, d1) -> (d0 * -16 + d1 * 32)>
#map45 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 31)>
#map46 = affine_map<(d0, d1) -> (42, d0 * 32, d1 * -32 + d0 * 32 + 51)>
#map47 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 83)>
#map48 = affine_map<(d0) -> (42, d0 * 32)>
#map49 = affine_map<(d0) -> (d0 - 41)>
#map50 = affine_map<(d0) -> ((d0 + 1) ceildiv 2)>
#map51 = affine_map<(d0) -> ((d0 * 16 + 39) floordiv 32 + 1)>
#map52 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * 16 + 40)>
#map53 = affine_map<(d0) -> (50, d0 * 32 + 32)>
#map54 = affine_map<(d0) -> (3, d0 * 32)>
#map55 = affine_map<(d0) -> (11, d0 * 32 + 32)>
#map56 = affine_map<(d0) -> (d0 - 2)>
#map57 = affine_map<()[s0] -> (s0 - 1)>
#map58 = affine_map<(d0)[s0] -> ((s0 + 38) floordiv 32 + 1, (d0 * 16 + s0 + 13) floordiv 32 + 1, (d0 * 32 + s0 + 26) floordiv 32 + 1)>
#map59 = affine_map<()[s0] -> ((s0 + 28) floordiv 32 + 1)>
#map60 = affine_map<(d0)[s0] -> (s0 + 29, d0 * 32 + 32)>
#map61 = affine_map<()[s0] -> (s0 - 2)>
#map62 = affine_map<(d0, d1)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32)>
#map63 = affine_map<(d0, d1)[s0] -> ((s0 + 38) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1)>
#map64 = affine_map<(d0, d1, d2)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32, (d2 * 32 - s0 - 27) ceildiv 32)>
#map65 = affine_map<(d0, d1, d2)[s0] -> ((s0 + 38) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1, (d2 * 32 + s0 + 27) floordiv 32 + 1)>
#map66 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 - s0 + 3)>
#map67 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - s0 + 3, d2 * -32 + d0 * 32 + d1 * 64 - s0 * 2 - 27)>
#map68 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 + 5)>
#map69 = affine_map<(d0)[s0] -> (s0 + 32, d0 * 32 + 32)>
#map70 = affine_map<(d0, d1, d2, d3)[s0] -> (1, (d0 * 32 - s0 + 2) ceildiv 2, (d1 * 32 - s0 + 2) ceildiv 2, (d2 * 32 - s0 + 2) ceildiv 2, d3 * 8, d3 * 16 - d0 * 16 + 1)>
#map71 = affine_map<(d0, d1, d2, d3)[s0] -> (21, d0 * 16 - d1 * 16 + s0 floordiv 2 + 15, d0 * 8 + 16, d1 * 16 + 15, d2 * 16 + 15, d3 * 16 + 15)>
#map72 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0 - 1)>
#map73 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 4 + 1, d2 * 2 + s0 - 1)>
#map74 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0)>
#map75 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 2 + s0, d2 * -32 + d0 * 32 + d1 * 4 + 3)>
#map76 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - d2 * 32 + s0 + 30)>
#map77 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 * 32 + s0 * 2 + 28)>
#map78 = affine_map<(d0, d1)[s0] -> (s0 + 29, d0 * 32 + 32, d1 * -32 + d0 * 32 + 61)>
#map79 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 29) floordiv 32 + 1)>
#map80 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 30)>
#map81 = affine_map<()[s0] -> ((s0 + 39) floordiv 32 + 1)>
#map82 = affine_map<(d0)[s0] -> (s0 + 40, d0 * 32 + 32)>
#map83 = affine_map<()[s0] -> ((s0 - 5) floordiv 32 + 1)>
#map84 = affine_map<(d0)[s0] -> (s0 + 1, d0 * 32 + 32)>
#set0 = affine_set<(d0, d1) : (d0 - 1 == 0, d1 * 32 - 38 == 0)>
#set1 = affine_set<() : (6 == 0)>
#set2 = affine_set<(d0, d1, d2) : (-d0 + 1 >= 0, -d1 >= 0, d2 * 32 - 38 == 0)>
#set3 = affine_set<(d0, d1) : (d1 * 2 - d0 - 1 >= 0, d1 - 1 >= 0)>
#set4 = affine_set<() : (1 == 0)>
#set5 = affine_set<(d0, d1, d2) : (d1 * 2 - d0 - 1 >= 0, d2 + d1 - d0 - 1 >= 0, d1 - d2 - 1 >= 0, d1 - 1 >= 0)>
#set6 = affine_set<(d0, d1, d2, d3) : (d1 * 2 - d0 - 1 >= 0, d2 + d1 - d0 - 1 >= 0, d1 - d2 - 1 >= 0, d1 - d3 - 1 >= 0, d1 - 1 >= 0)>
#set7 = affine_set<(d0, d1, d2, d3) : (d0 - 2 == 0, d1 - 1 == 0, -d2 + 1 >= 0, -d3 + 1 >= 0)>
#set8 = affine_set<(d0, d1, d2) : (d0 - (d1 * 16 + d2 - 16) ceildiv 16 >= 0, d2 floordiv 16 - d1 >= 0)>
#set9 = affine_set<(d0, d1) : (d1 floordiv 16 - d0 >= 0)>
#set10 = affine_set<(d0, d1) : (d0 - (d1 * 2 - 22) ceildiv 32 >= 0)>
#set11 = affine_set<(d0, d1, d2) : ((d1 * 32 + d2 * 2 - 9) floordiv 32 - d0 >= 0, d1 - (d2 * 2 - 22) ceildiv 32 >= 0)>
#set12 = affine_set<(d0, d1, d2, d3) : (d1 * 2 - d0 - 1 >= 0, d1 - d0 >= 0, d1 + d2 - d0 - 1 >= 0, d1 + d3 - d0 - 1 >= 0)>
#set13 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, d0 - (d1 * 32 - 9) ceildiv 32 >= 0, d2 == 0, d3 - 1 >= 0)>
#set14 = affine_set<(d0, d1, d2) : (d0 >= 0, d0 - (d1 * 32 - 9) ceildiv 32 >= 0, d2 == 0)>
#set15 = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 - 1 >= 0, d3 - 1 >= 0)>
#set16 = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 - 1 >= 0)>
#set17 = affine_set<(d0, d1) : (d0 >= 0, d1 == 0)>
#set18 = affine_set<(d0, d1, d2) : (-d0 >= 0, d1 * 2 - d0 - 1 >= 0, d2 * 2 - d0 - 1 >= 0)>
#set19 = affine_set<(d0) : ((d0 * 16 + 39) mod 32 == 0)>
#set20 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set21 = affine_set<() : (17 == 0)>
#set22 = affine_set<(d0, d1) : (d0 == 0, d1 == 0)>
#set23 = affine_set<(d0, d1) : (-d0 >= 0, d1 * 2 - d0 - 1 >= 0)>
#set24 = affine_set<(d0) : (d0 - 1 >= 0)>
#set25 = affine_set<(d0) : (d0 + 1 == 0)>
#set26 = affine_set<() : (5 == 0)>
#set27 = affine_set<(d0) : (d0 - 2 == 0)>
#set28 = affine_set<(d0, d1)[s0] : (d0 - 1 == 0, d1 * 32 - (s0 + 28) == 0)>
#set29 = affine_set<()[s0] : ((s0 + 28) mod 32 == 0)>
#set30 = affine_set<(d0, d1, d2)[s0] : (-d0 + 1 >= 0, -d1 + (s0 - 4) floordiv 32 >= 0, d2 * 32 - (s0 + 28) == 0)>
#set31 = affine_set<(d0, d1)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set32 = affine_set<()[s0] : ((s0 + 1) mod 2 == 0)>
#set33 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set34 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - d3 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set35 = affine_set<(d0, d1, d2, d3)[s0] : (d0 - 2 == 0, d1 - 1 == 0, -d2 + (s0 + 31) floordiv 32 >= 0, -d3 + (s0 + 31) floordiv 32 >= 0)>
#set36 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set37 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 + d2 * 2 - s0 + 1) floordiv 32 - d0 >= 0, d1 - (d2 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set38 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d1 * 32 - s0 + 11) floordiv 32 - d0 >= 0, (d1 * 32 + d2 * 32 - s0 - 1) floordiv 32 - d0 >= 0, (d1 * 32 + d3 * 32 - s0 - 1) floordiv 32 - d0 >= 0)>
#set39 = affine_set<(d0, d1, d2, d3)[s0] : (d0 >= 0, d0 - (d1 * 32 - s0 + 1) ceildiv 32 >= 0, d2 == 0, d3 - 1 >= 0)>
#set40 = affine_set<(d0, d1, d2)[s0] : (d0 >= 0, d0 - (d1 * 32 - s0 + 1) ceildiv 32 >= 0, d2 == 0)>
#set41 = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 - 1 >= 0, d3 - 1 >= 0)>
#set42 = affine_set<(d0)[s0] : ((d0 * 16 + s0 + 29) mod 32 == 0)>
#set43 = affine_set<()[s0] : ((s0 + 7) mod 32 == 0)>
#set44 = affine_set<()[s0] : ((s0 + 27) mod 32 == 0)>
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
    %c0_i32 = constant 0 : i32
    %c10_i32 = constant 10 : i32
    %c1_i32 = constant 1 : i32
    %c8 = constant 8 : index
    %c10 = constant 10 : index
    %c16 = constant 16 : index
    %c15 = constant 15 : index
    %c1 = constant 1 : index
    %c20 = constant 20 : index
    %0 = alloc() : memref<10x10x10xf64>
    %1 = alloc() : memref<10x10x10xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%2: i32):  // 2 preds: ^bb0, ^bb4
    %3 = cmpi "slt", %2, %c10_i32 : i32
    %4 = index_cast %2 : i32 to index
    cond_br %3, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    affine.for %arg2 = -1 to 3 {
      affine.for %arg3 = #map0(%arg2) to min #map1(%arg2) {
        affine.if #set0(%arg2, %arg3) {
          affine.for %arg4 = 0 to 2 {
            affine.for %arg5 = max #map2(%arg4) to min #map3(%arg4) {
              affine.if #set1() {
                call @S0(%1, %c15, %c8, %c1, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.for %arg4 = max #map4(%arg2, %arg3) to min #map5(%arg2, %arg3) {
          affine.if #set2(%arg2, %arg3, %arg4) {
            affine.for %arg5 = max #map6(%arg2, %arg3) to min #map7(%arg2, %arg3) {
              affine.if #set1() {
                %23 = affine.apply #map8(%arg5)
                call @S0(%1, %c15, %23, %c8, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set0(%arg2, %arg3) {
            affine.for %arg5 = #map9(%arg4) to min #map3(%arg4) {
              affine.if #set1() {
                %23 = affine.apply #map8(%arg5)
                call @S0(%1, %c15, %c8, %23, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.for %arg5 = max #map10(%arg2, %arg3, %arg4) to min #map11(%arg2, %arg3, %arg4) {
            affine.if #set3(%arg2, %arg3) {
              affine.if #set4() {
                affine.for %arg6 = max #map12(%arg3, %arg4) to min #map13(%arg3, %arg4) {
                  affine.for %arg7 = max #map12(%arg3, %arg5) to min #map13(%arg3, %arg5) {
                    %23 = affine.apply #map14(%arg3)[%c10]
                    %24 = affine.apply #map15(%arg3, %arg6)[%c10]
                    call @S1(%0, %23, %c8, %24, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set5(%arg2, %arg3, %arg4) {
              affine.if #set4() {
                affine.for %arg6 = max #map16(%arg2, %arg3, %arg4) to min #map17(%arg2, %arg3, %arg4) {
                  affine.for %arg7 = max #map12(%arg4, %arg5) to min #map13(%arg4, %arg5) {
                    %23 = affine.apply #map14(%arg4)[%c10]
                    %24 = affine.apply #map15(%arg4, %arg6)[%c10]
                    call @S1(%0, %23, %24, %c8, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set6(%arg2, %arg3, %arg4, %arg5) {
              affine.if #set4() {
                affine.for %arg6 = max #map16(%arg2, %arg3, %arg5) to min #map17(%arg2, %arg3, %arg5) {
                  affine.for %arg7 = max #map12(%arg4, %arg5) to #map18(%arg4) {
                    %23 = affine.apply #map14(%arg5)[%c10]
                    %24 = affine.apply #map15(%arg5, %arg6)[%c10]
                    %25 = affine.apply #map15(%arg5, %arg7)[%c10]
                    call @S1(%0, %23, %24, %25, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set7(%arg2, %arg3, %arg4, %arg5) {
              affine.for %arg6 = max #map19(%arg4) to min #map20(%arg4) {
                affine.for %arg7 = max #map19(%arg5) to min #map20(%arg5) {
                  %23 = affine.apply #map21(%arg6)
                  call @S1(%0, %c16, %c1, %23, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.for %arg6 = max #map22(%arg2, %arg3, %arg4, %arg5) to min #map23(%arg2, %arg3, %arg4, %arg5) {
              affine.if #set8(%arg2, %arg3, %arg6) {
                affine.for %arg7 = max #map24(%arg4, %arg6) to min #map25(%arg4, %arg6) {
                  affine.for %arg8 = max #map24(%arg5, %arg6) to min #map25(%arg5, %arg6) {
                    %23 = affine.apply #map26(%arg6, %arg7)
                    call @S0(%1, %arg6, %c1, %23, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg7 = max #map27(%arg2, %arg3, %arg6) to #map28(%arg2, %arg3, %arg6) {
                affine.for %arg8 = max #map24(%arg4, %arg6) to min #map25(%arg4, %arg6) {
                  affine.for %arg9 = max #map24(%arg5, %arg6) to min #map25(%arg5, %arg6) {
                    %23 = affine.apply #map26(%arg6, %arg7)
                    %24 = affine.apply #map26(%arg6, %arg8)
                    call @S0(%1, %arg6, %23, %24, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg7 = max #map29(%arg2, %arg3, %arg6) to min #map30(%arg2, %arg3, %arg6) {
                affine.if #set9(%arg4, %arg6) {
                  affine.for %arg8 = max #map24(%arg5, %arg6) to min #map25(%arg5, %arg6) {
                    %23 = affine.apply #map26(%arg6, %arg7)
                    call @S0(%1, %arg6, %23, %c1, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
                affine.for %arg8 = max #map31(%arg4, %arg6) to min #map25(%arg4, %arg6) {
                  affine.if #set9(%arg5, %arg6) {
                    %23 = affine.apply #map26(%arg6, %arg7)
                    %24 = affine.apply #map26(%arg6, %arg8)
                    call @S0(%1, %arg6, %23, %24, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                  affine.for %arg9 = max #map31(%arg5, %arg6) to min #map25(%arg5, %arg6) {
                    %23 = affine.apply #map32(%arg6, %arg7)
                    %24 = affine.apply #map32(%arg6, %arg8)
                    call @S1(%0, %arg6, %23, %24, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                    %25 = affine.apply #map26(%arg6, %arg7)
                    %26 = affine.apply #map26(%arg6, %arg8)
                    call @S0(%1, %arg6, %25, %26, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                  affine.if #set10(%arg5, %arg6) {
                    %23 = affine.apply #map32(%arg6, %arg7)
                    %24 = affine.apply #map32(%arg6, %arg8)
                    call @S1(%0, %arg6, %23, %24, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
                affine.if #set10(%arg4, %arg6) {
                  affine.for %arg8 = max #map31(%arg5, %arg6) to min #map33(%arg5, %arg6) {
                    %23 = affine.apply #map32(%arg6, %arg7)
                    call @S1(%0, %arg6, %23, %c8, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg7 = #map34(%arg2, %arg3, %arg6) to min #map35(%arg2, %arg3, %arg6) {
                affine.for %arg8 = max #map31(%arg4, %arg6) to min #map33(%arg4, %arg6) {
                  affine.for %arg9 = max #map31(%arg5, %arg6) to min #map33(%arg5, %arg6) {
                    %23 = affine.apply #map32(%arg6, %arg7)
                    %24 = affine.apply #map32(%arg6, %arg8)
                    call @S1(%0, %arg6, %23, %24, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.if #set11(%arg2, %arg3, %arg6) {
                affine.for %arg7 = max #map31(%arg4, %arg6) to min #map33(%arg4, %arg6) {
                  affine.for %arg8 = max #map31(%arg5, %arg6) to min #map33(%arg5, %arg6) {
                    %23 = affine.apply #map32(%arg6, %arg7)
                    call @S1(%0, %arg6, %c8, %23, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set12(%arg2, %arg3, %arg4, %arg5) {
              affine.if #set4() {
                affine.for %arg6 = max #map36(%arg2, %arg3, %arg4) to min #map37(%arg2, %arg3, %arg4) {
                  affine.for %arg7 = max #map36(%arg2, %arg3, %arg5) to min #map37(%arg2, %arg3, %arg5) {
                    %23 = affine.apply #map38(%arg2, %arg3)[%c10]
                    %24 = affine.apply #map39(%arg2, %arg3, %arg6)[%c10]
                    call @S0(%1, %23, %c8, %24, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set13(%arg2, %arg3, %arg4, %arg5) {
              affine.for %arg6 = max #map40(%arg2, %arg3) to min #map41(%arg2, %arg3) {
                affine.for %arg7 = #map9(%arg5) to min #map3(%arg5) {
                  %23 = affine.apply #map8(%arg6)
                  call @S0(%1, %c15, %23, %c1, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.if #set14(%arg2, %arg3, %arg5) {
              affine.for %arg6 = max #map40(%arg2, %arg3) to min #map41(%arg2, %arg3) {
                affine.for %arg7 = max #map2(%arg4) to min #map3(%arg4) {
                  %23 = affine.apply #map8(%arg6)
                  %24 = affine.apply #map8(%arg7)
                  call @S0(%1, %c15, %23, %24, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.if #set15(%arg2, %arg3, %arg4, %arg5) {
              affine.for %arg6 = #map9(%arg4) to min #map3(%arg4) {
                affine.for %arg7 = #map9(%arg5) to min #map3(%arg5) {
                  %23 = affine.apply #map8(%arg6)
                  call @S0(%1, %c15, %c1, %23, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
          affine.if #set16(%arg2, %arg3, %arg4) {
            affine.if #set1() {
              affine.for %arg5 = #map9(%arg4) to #map18(%arg4) {
                %23 = affine.apply #map8(%arg5)
                call @S0(%1, %c15, %c1, %23, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set17(%arg2, %arg4) {
            affine.if #set1() {
              affine.for %arg5 = max #map40(%arg2, %arg3) to min #map7(%arg2, %arg3) {
                %23 = affine.apply #map8(%arg5)
                call @S0(%1, %c15, %23, %c1, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set18(%arg2, %arg3, %arg4) {
            affine.if #set19(%arg2) {
              affine.for %arg5 = max #map42(%arg2, %arg4) to #map18(%arg4) {
                %23 = affine.apply #map43(%arg2)
                %24 = affine.apply #map44(%arg2, %arg3)
                %25 = affine.apply #map45(%arg2, %arg5)
                call @S1(%0, %23, %24, %25, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set20(%arg2, %arg4) {
            affine.if #set21() {
              affine.for %arg5 = max #map46(%arg2, %arg3) to min #map47(%arg2, %arg3) {
                affine.for %arg6 = max #map48(%arg4) to #map18(%arg4) {
                  %23 = affine.apply #map49(%arg5)
                  %24 = affine.apply #map49(%arg6)
                  call @S1(%0, %c20, %23, %24, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set22(%arg2, %arg3) {
          affine.if #set1() {
            affine.for %arg4 = 0 to 2 {
              affine.for %arg5 = max #map2(%arg4) to min #map3(%arg4) {
                call @S0(%1, %c15, %c1, %c8, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.if #set23(%arg2, %arg3) {
          affine.if #set19(%arg2) {
            affine.for %arg4 = #map50(%arg2) to #map51(%arg2) {
              affine.for %arg5 = max #map42(%arg2, %arg4) to min #map52(%arg2, %arg4) {
                %23 = affine.apply #map43(%arg2)
                %24 = affine.apply #map44(%arg2, %arg3)
                call @S1(%0, %23, %24, %c8, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.if #set24(%arg2) {
          affine.if #set21() {
            affine.for %arg4 = 1 to 2 {
              affine.for %arg5 = max #map46(%arg2, %arg3) to min #map47(%arg2, %arg3) {
                affine.for %arg6 = max #map48(%arg4) to min #map53(%arg4) {
                  %23 = affine.apply #map49(%arg5)
                  call @S1(%0, %c20, %23, %c8, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set25(%arg2) {
        affine.if #set26() {
          affine.for %arg3 = 0 to 1 {
            affine.for %arg4 = 0 to 1 {
              affine.for %arg5 = max #map54(%arg3) to min #map55(%arg3) {
                affine.for %arg6 = max #map54(%arg4) to min #map55(%arg4) {
                  %23 = affine.apply #map56(%arg5)
                  call @S0(%1, %c1, %c8, %23, %0) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set27(%arg2) {
        affine.if #set21() {
          affine.for %arg3 = 1 to 2 {
            affine.for %arg4 = 1 to 2 {
              affine.for %arg5 = max #map48(%arg3) to min #map53(%arg3) {
                affine.for %arg6 = max #map48(%arg4) to min #map53(%arg4) {
                  %23 = affine.apply #map49(%arg5)
                  call @S1(%0, %c20, %c8, %23, %1) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
    }
    call @print_array(%c10_i32, %0) : (i32, memref<10x10x10xf64>) -> ()
    return %c0_i32 : i32
  ^bb3(%5: i32):  // 2 preds: ^bb1, ^bb7
    %6 = cmpi "slt", %5, %c10_i32 : i32
    %7 = index_cast %5 : i32 to index
    cond_br %6, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %8 = addi %2, %c1_i32 : i32
    br ^bb1(%8 : i32)
  ^bb5(%9: i32):  // 2 preds: ^bb3, ^bb6
    %10 = cmpi "slt", %9, %c10_i32 : i32
    %11 = index_cast %9 : i32 to index
    cond_br %10, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %12 = addi %2, %5 : i32
    %13 = subi %c10_i32, %9 : i32
    %14 = addi %12, %13 : i32
    %15 = sitofp %14 : i32 to f64
    %16 = sitofp %c10_i32 : i32 to f64
    %17 = mulf %15, %16 : f64
    %18 = sitofp %c10_i32 : i32 to f64
    %19 = divf %17, %18 : f64
    store %19, %1[%4, %7, %11] : memref<10x10x10xf64>
    %20 = load %1[%4, %7, %11] : memref<10x10x10xf64>
    store %20, %0[%4, %7, %11] : memref<10x10x10xf64>
    %21 = addi %9, %c1_i32 : i32
    br ^bb5(%21 : i32)
  ^bb7:  // pred: ^bb5
    %22 = addi %5, %c1_i32 : i32
    br ^bb3(%22 : i32)
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
      affine.for %arg5 = 1 to #map57()[%0] {
        affine.for %arg6 = 1 to #map57()[%0] {
          affine.for %arg7 = 1 to #map57()[%0] {
            call @S0(%arg3, %arg5, %arg6, %arg7, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
          }
        }
      }
      affine.for %arg5 = 1 to #map57()[%0] {
        affine.for %arg6 = 1 to #map57()[%0] {
          affine.for %arg7 = 1 to #map57()[%0] {
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
      affine.for %arg5 = #map0(%arg4) to min #map58(%arg4)[%0] {
        affine.if #set28(%arg4, %arg5)[%0] {
          affine.for %arg6 = 0 to #map59()[%0] {
            affine.for %arg7 = max #map2(%arg6) to min #map60(%arg6)[%0] {
              affine.if #set29()[%0] {
                %1 = affine.apply #map61()[%0]
                call @S0(%arg3, %c15, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.for %arg6 = max #map62(%arg4, %arg5)[%0] to min #map63(%arg4, %arg5)[%0] {
          affine.if #set30(%arg4, %arg5, %arg6)[%0] {
            affine.for %arg7 = max #map6(%arg4, %arg5) to min #map7(%arg4, %arg5) {
              affine.if #set29()[%0] {
                %1 = affine.apply #map8(%arg7)
                %2 = affine.apply #map61()[%0]
                call @S0(%arg3, %c15, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set28(%arg4, %arg5)[%0] {
            affine.for %arg7 = #map9(%arg6) to min #map60(%arg6)[%0] {
              affine.if #set29()[%0] {
                %1 = affine.apply #map61()[%0]
                %2 = affine.apply #map8(%arg7)
                call @S0(%arg3, %c15, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.for %arg7 = max #map64(%arg4, %arg5, %arg6)[%0] to min #map65(%arg4, %arg5, %arg6)[%0] {
            affine.if #set31(%arg4, %arg5)[%0] {
              affine.if #set32()[%0] {
                affine.for %arg8 = max #map66(%arg5, %arg6)[%0] to min #map13(%arg5, %arg6) {
                  affine.for %arg9 = max #map66(%arg5, %arg7)[%0] to min #map13(%arg5, %arg7) {
                    %1 = affine.apply #map14(%arg5)[%0]
                    %2 = affine.apply #map61()[%0]
                    %3 = affine.apply #map15(%arg5, %arg8)[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set33(%arg4, %arg5, %arg6)[%0] {
              affine.if #set32()[%0] {
                affine.for %arg8 = max #map67(%arg4, %arg5, %arg6)[%0] to min #map68(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map66(%arg6, %arg7)[%0] to min #map13(%arg6, %arg7) {
                    %1 = affine.apply #map14(%arg6)[%0]
                    %2 = affine.apply #map15(%arg6, %arg8)[%0]
                    %3 = affine.apply #map61()[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set34(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set32()[%0] {
                affine.for %arg8 = max #map67(%arg4, %arg5, %arg7)[%0] to min #map68(%arg4, %arg5, %arg7)[%0] {
                  affine.for %arg9 = max #map66(%arg6, %arg7)[%0] to #map18(%arg6) {
                    %1 = affine.apply #map14(%arg7)[%0]
                    %2 = affine.apply #map15(%arg7, %arg8)[%0]
                    %3 = affine.apply #map15(%arg7, %arg9)[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set35(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map19(%arg6) to min #map69(%arg6)[%0] {
                affine.for %arg9 = max #map19(%arg7) to min #map69(%arg7)[%0] {
                  %1 = affine.apply #map21(%arg8)
                  call @S1(%arg2, %c16, %c1, %1, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.for %arg8 = max #map70(%arg4, %arg5, %arg6, %arg7)[%0] to min #map71(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set8(%arg4, %arg5, %arg8) {
                affine.for %arg9 = max #map24(%arg6, %arg8) to min #map72(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map24(%arg7, %arg8) to min #map72(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map26(%arg8, %arg9)
                    call @S0(%arg3, %arg8, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map27(%arg4, %arg5, %arg8) to #map28(%arg4, %arg5, %arg8) {
                affine.for %arg10 = max #map24(%arg6, %arg8) to min #map72(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map24(%arg7, %arg8) to min #map72(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map26(%arg8, %arg9)
                    %2 = affine.apply #map26(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map29(%arg4, %arg5, %arg8) to min #map73(%arg4, %arg5, %arg8)[%0] {
                affine.if #set9(%arg6, %arg8) {
                  affine.for %arg10 = max #map24(%arg7, %arg8) to min #map72(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map26(%arg8, %arg9)
                    call @S0(%arg3, %arg8, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
                affine.for %arg10 = max #map31(%arg6, %arg8) to min #map72(%arg6, %arg8)[%0] {
                  affine.if #set9(%arg7, %arg8) {
                    %1 = affine.apply #map26(%arg8, %arg9)
                    %2 = affine.apply #map26(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                  affine.for %arg11 = max #map31(%arg7, %arg8) to min #map72(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    %2 = affine.apply #map32(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                    %3 = affine.apply #map26(%arg8, %arg9)
                    %4 = affine.apply #map26(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %3, %4, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                  affine.if #set36(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    %2 = affine.apply #map32(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
                affine.if #set36(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map31(%arg7, %arg8) to min #map74(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    %2 = affine.apply #map61()[%0]
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = #map34(%arg4, %arg5, %arg8) to min #map75(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg10 = max #map31(%arg6, %arg8) to min #map74(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map31(%arg7, %arg8) to min #map74(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    %2 = affine.apply #map32(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
              affine.if #set37(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg9 = max #map31(%arg6, %arg8) to min #map74(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map31(%arg7, %arg8) to min #map74(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map61()[%0]
                    %2 = affine.apply #map32(%arg8, %arg9)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set38(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set32()[%0] {
                affine.for %arg8 = max #map76(%arg4, %arg5, %arg6)[%0] to min #map77(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map76(%arg4, %arg5, %arg7)[%0] to min #map77(%arg4, %arg5, %arg7)[%0] {
                    %1 = affine.apply #map38(%arg4, %arg5)[%0]
                    %2 = affine.apply #map61()[%0]
                    %3 = affine.apply #map39(%arg4, %arg5, %arg8)[%0]
                    call @S0(%arg3, %1, %2, %3, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set39(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map40(%arg4, %arg5) to min #map78(%arg4, %arg5)[%0] {
                affine.for %arg9 = #map9(%arg7) to min #map60(%arg7)[%0] {
                  %1 = affine.apply #map8(%arg8)
                  call @S0(%arg3, %c15, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.if #set40(%arg4, %arg5, %arg7)[%0] {
              affine.for %arg8 = max #map40(%arg4, %arg5) to min #map78(%arg4, %arg5)[%0] {
                affine.for %arg9 = max #map2(%arg6) to min #map60(%arg6)[%0] {
                  %1 = affine.apply #map8(%arg8)
                  %2 = affine.apply #map8(%arg9)
                  call @S0(%arg3, %c15, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
            affine.if #set41(%arg4, %arg5, %arg6, %arg7) {
              affine.for %arg8 = #map9(%arg6) to min #map60(%arg6)[%0] {
                affine.for %arg9 = #map9(%arg7) to min #map60(%arg7)[%0] {
                  %1 = affine.apply #map8(%arg8)
                  call @S0(%arg3, %c15, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
          affine.if #set16(%arg4, %arg5, %arg6) {
            affine.if #set29()[%0] {
              affine.for %arg7 = #map9(%arg6) to #map18(%arg6) {
                %1 = affine.apply #map8(%arg7)
                call @S0(%arg3, %c15, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set17(%arg4, %arg6) {
            affine.if #set29()[%0] {
              affine.for %arg7 = max #map40(%arg4, %arg5) to min #map7(%arg4, %arg5) {
                %1 = affine.apply #map8(%arg7)
                call @S0(%arg3, %c15, %1, %c1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set18(%arg4, %arg5, %arg6) {
            affine.if #set42(%arg4)[%0] {
              affine.for %arg7 = max #map42(%arg4, %arg6) to #map18(%arg6) {
                %1 = affine.apply #map43(%arg4)
                %2 = affine.apply #map44(%arg4, %arg5)
                %3 = affine.apply #map45(%arg4, %arg7)
                call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
          affine.if #set20(%arg4, %arg6) {
            affine.if #set43()[%0] {
              affine.for %arg7 = max #map46(%arg4, %arg5) to min #map47(%arg4, %arg5) {
                affine.for %arg8 = max #map48(%arg6) to #map18(%arg6) {
                  %1 = affine.apply #map49(%arg7)
                  %2 = affine.apply #map49(%arg8)
                  call @S1(%arg2, %c20, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set22(%arg4, %arg5) {
          affine.if #set29()[%0] {
            affine.for %arg6 = 0 to #map59()[%0] {
              affine.for %arg7 = max #map2(%arg6) to min #map60(%arg6)[%0] {
                %1 = affine.apply #map61()[%0]
                call @S0(%arg3, %c15, %c1, %1, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.if #set23(%arg4, %arg5) {
          affine.if #set42(%arg4)[%0] {
            affine.for %arg6 = #map50(%arg4) to #map79(%arg4)[%0] {
              affine.for %arg7 = max #map42(%arg4, %arg6) to min #map80(%arg4, %arg6)[%0] {
                %1 = affine.apply #map43(%arg4)
                %2 = affine.apply #map44(%arg4, %arg5)
                %3 = affine.apply #map61()[%0]
                call @S1(%arg2, %1, %2, %3, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
              }
            }
          }
        }
        affine.if #set24(%arg4) {
          affine.if #set43()[%0] {
            affine.for %arg6 = 1 to #map81()[%0] {
              affine.for %arg7 = max #map46(%arg4, %arg5) to min #map47(%arg4, %arg5) {
                affine.for %arg8 = max #map48(%arg6) to min #map82(%arg6)[%0] {
                  %1 = affine.apply #map49(%arg7)
                  %2 = affine.apply #map61()[%0]
                  call @S1(%arg2, %c20, %1, %2, %arg3) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set25(%arg4) {
        affine.if #set44()[%0] {
          affine.for %arg5 = 0 to #map83()[%0] {
            affine.for %arg6 = 0 to #map83()[%0] {
              affine.for %arg7 = max #map54(%arg5) to min #map84(%arg5)[%0] {
                affine.for %arg8 = max #map54(%arg6) to min #map84(%arg6)[%0] {
                  %1 = affine.apply #map61()[%0]
                  %2 = affine.apply #map56(%arg7)
                  call @S0(%arg3, %c1, %1, %2, %arg2) : (memref<10x10x10xf64>, index, index, index, memref<10x10x10xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set27(%arg4) {
        affine.if #set43()[%0] {
          affine.for %arg5 = 1 to #map81()[%0] {
            affine.for %arg6 = 1 to #map81()[%0] {
              affine.for %arg7 = max #map48(%arg5) to min #map82(%arg5)[%0] {
                affine.for %arg8 = max #map48(%arg6) to min #map82(%arg6)[%0] {
                  %1 = affine.apply #map61()[%0]
                  %2 = affine.apply #map49(%arg7)
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

