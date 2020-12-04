#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 ceildiv 2)>
#map2 = affine_map<(d0)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 13) floordiv 32 + 1, (d0 * 32 + s0 + 26) floordiv 32 + 1)>
#map3 = affine_map<(d0) -> ((d0 - 1) ceildiv 2)>
#map4 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 12) floordiv 32 + 1)>
#map5 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 15)>
#map6 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 13)>
#map7 = affine_map<(d0) -> (d0 * 8 + 7)>
#map8 = affine_map<()[s0] -> (s0 - 2)>
#map9 = affine_map<(d0, d1)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32)>
#map10 = affine_map<(d0, d1)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1)>
#map11 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 - 27)>
#map12 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 + 5)>
#map13 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 2) ceildiv 2)>
#map14 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 + s0 - 2)>
#map15 = affine_map<(d0) -> (d0 * 32)>
#map16 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 14)>
#map17 = affine_map<(d0, d1, d2)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32, (d2 * 32 - s0 - 27) ceildiv 32)>
#map18 = affine_map<(d0, d1, d2)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1, (d2 * 32 + s0 + 27) floordiv 32 + 1)>
#map19 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 - s0 + 3)>
#map20 = affine_map<(d0, d1) -> (d0 * 32 + 1, d1 * 32 + 32)>
#map21 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 1) ceildiv 2)>
#map22 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - s0 + 3, d2 * -32 + d0 * 32 + d1 * 64 - s0 * 2 - 27)>
#map23 = affine_map<(d0) -> (d0 * 32 + 32)>
#map24 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 2)>
#map25 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0)>
#map26 = affine_map<(d0) -> (d0 * 8)>
#map27 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 1)>
#map28 = affine_map<(d0, d1, d2, d3)[s0] -> (1, (d0 * 32 - s0 + 2) ceildiv 2, (d1 * 32 - s0 + 2) ceildiv 2, (d2 * 32 - s0 + 2) ceildiv 2, d3 * 8, d3 * 16 - d0 * 16 + 1)>
#map29 = affine_map<(d0, d1, d2, d3)[s0] -> (501, d0 * 16 - d1 * 16 + s0 floordiv 2 + 15, d0 * 8 + 16, d1 * 16 + 15, d2 * 16 + 15, d3 * 16 + 15)>
#map30 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 1)>
#map31 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0 - 1)>
#map32 = affine_map<(d0, d1) -> (d0 * -2 + d1)>
#map33 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 31)>
#map34 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 - 29)>
#map35 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 29)>
#map36 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 4 + 1, d2 * 2 + s0 - 1)>
#map37 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
#map38 = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
#map39 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0)>
#map40 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 + 1)>
#map41 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 2 + s0, d2 * -32 + d0 * 32 + d1 * 4 + 3)>
#map42 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - d2 * 32 + s0 + 30)>
#map43 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 * 32 + s0 * 2 + 28)>
#map44 = affine_map<(d0, d1)[s0] -> ((d0 * 32 - d1 * 32 + s0 + 29) ceildiv 2)>
#map45 = affine_map<(d0, d1, d2)[s0] -> (d0 * -32 + d1 * 32 + d2 - s0 - 29)>
#map46 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 + 31, d2 * -32 + d0 * 32 + d1 * 64 + 29)>
#map47 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29, d2 * -32 + d0 * 32 + d1 * 64 + 61)>
#map48 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29)>
#map49 = affine_map<(d0) -> (d0 * 16 + 15)>
#map50 = affine_map<(d0, d1) -> (d0 * -32 + d1 - 30)>
#map51 = affine_map<(d0, d1) -> (d0 * 32, d1 * 32 + 31)>
#map52 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 29)>
#map53 = affine_map<(d0) -> (d0 * 8 + 15)>
#map54 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 30)>
#map55 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 + 61)>
#map56 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 32)>
#map57 = affine_map<(d0, d1) -> (d0 * -16 + d1 * 32)>
#map58 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 31)>
#map59 = affine_map<(d0, d1) -> (1002, d0 * 32, d1 * -32 + d0 * 32 + 1971)>
#map60 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 2003)>
#map61 = affine_map<(d0) -> (1002, d0 * 32)>
#map62 = affine_map<(d0) -> (d0 - 1001)>
#map63 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 28) floordiv 32 + 1)>
#map64 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 31)>
#map65 = affine_map<(d0) -> ((d0 + 1) ceildiv 2)>
#map66 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 29) floordiv 32 + 1)>
#map67 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 30)>
#map68 = affine_map<()[s0] -> ((s0 + 999) floordiv 32 + 1)>
#map69 = affine_map<(d0)[s0] -> (s0 + 1000, d0 * 32 + 32)>
#map70 = affine_map<()[s0] -> ((s0 - 5) floordiv 32 + 1)>
#map71 = affine_map<(d0) -> (3, d0 * 32)>
#map72 = affine_map<(d0)[s0] -> (s0 + 1, d0 * 32 + 32)>
#map73 = affine_map<(d0) -> (d0 - 2)>
#set0 = affine_set<(d0, d1)[s0] : (d0 * 16 - (d1 * 32 - s0 - 12) == 0)>
#set1 = affine_set<(d0) : ((d0 + 1) mod 2 == 0)>
#set2 = affine_set<(d0)[s0] : ((d0 * 16 + s0 * 31 + 20) mod 32 == 0)>
#set3 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 - s0 - 12) floordiv 16 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - (s0 + 28) ceildiv 32 >= 0)>
#set4 = affine_set<()[s0] : ((s0 * 31 + 4) mod 32 == 0)>
#set5 = affine_set<(d0, d1)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set6 = affine_set<()[s0] : ((s0 + 1) mod 2 == 0)>
#set7 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set8 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - d3 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set9 = affine_set<(d0, d1, d2, d3)[s0] : (d0 - d1 * 2 == 0, d0 - 2 >= 0, d0 - (d2 * 32 - s0 + 1) ceildiv 16 >= 0, d0 - (d3 * 32 - s0 + 1) ceildiv 16 >= 0)>
#set10 = affine_set<(d0) : (d0 mod 2 == 0)>
#set11 = affine_set<(d0, d1, d2) : (d0 - (d1 * 16 + d2 - 16) ceildiv 16 >= 0, d2 floordiv 16 - d1 >= 0)>
#set12 = affine_set<(d0, d1) : (d1 floordiv 16 - d0 >= 0)>
#set13 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set14 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 + d2 * 2 - s0 + 1) floordiv 32 - d0 >= 0, d1 - (d2 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set15 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d1 * 32 - s0 + 971) floordiv 32 - d0 >= 0, (d1 * 32 + d2 * 32 - s0 - 1) floordiv 32 - d0 >= 0, (d1 * 32 + d3 * 32 - s0 - 1) floordiv 32 - d0 >= 0)>
#set16 = affine_set<(d0, d1, d2, d3)[s0] : (d0 - (d1 * 32 + d2 * 32 - s0 + 1) ceildiv 32 >= 0, d0 - d2 * 2 >= 0, -d2 + 30 >= 0, d3 - d2 - 1 >= 0)>
#set17 = affine_set<(d0, d1, d2)[s0] : (d0 - (d1 * 32 + d2 * 32 - s0 + 1) ceildiv 32 >= 0, d0 - d2 * 2 >= 0, -d2 + 30 >= 0)>
#set18 = affine_set<(d0, d1, d2, d3) : (d0 - d1 * 2 == 0, -d0 + 60 >= 0, d2 * 2 - d0 - 2 >= 0, d3 * 2 - d0 - 2 >= 0)>
#set19 = affine_set<(d0, d1) : (d0 - d1 * 2 >= 0, -d1 + 30 >= 0)>
#set20 = affine_set<()[s0] : ((s0 + 28) mod 32 == 0)>
#set21 = affine_set<(d0, d1, d2) : (d0 - d1 * 2 == 0, -d0 + 60 >= 0, d2 * 2 - d0 - 2 >= 0)>
#set22 = affine_set<(d0)[s0] : ((d0 * 16 + s0 + 28) mod 32 == 0)>
#set23 = affine_set<(d0, d1, d2) : (-d0 + 60 >= 0, d1 * 2 - d0 - 1 >= 0, d2 * 2 - d0 - 1 >= 0)>
#set24 = affine_set<(d0)[s0] : ((d0 * 16 + s0 + 29) mod 32 == 0)>
#set25 = affine_set<(d0, d1) : (d0 - 61 >= 0, d1 - 31 >= 0)>
#set26 = affine_set<()[s0] : ((s0 + 7) mod 32 == 0)>
#set27 = affine_set<(d0, d1) : (-d0 + 60 >= 0, d0 - d1 * 2 == 0)>
#set28 = affine_set<(d0, d1) : (-d0 + 60 >= 0, d1 * 2 - d0 - 1 >= 0)>
#set29 = affine_set<(d0) : (d0 - 61 >= 0)>
#set30 = affine_set<(d0) : (d0 + 1 == 0)>
#set31 = affine_set<()[s0] : ((s0 + 27) mod 32 == 0)>
#set32 = affine_set<(d0) : (d0 - 62 == 0)>
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
  llvm.mlir.global internal constant @str3("A\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c120_i32 = constant 120 : i32
    %c500_i32 = constant 500 : i32
    %c42_i32 = constant 42 : i32
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<120x120x120xf64>
    %1 = alloc() : memref<120x120x120xf64>
    call @init_array(%c120_i32, %0, %1) : (i32, memref<120x120x120xf64>, memref<120x120x120xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    call @kernel_heat_3d_new(%c500_i32, %c120_i32, %0, %1) : (i32, i32, memref<120x120x120xf64>, memref<120x120x120xf64>) -> ()
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
      call @print_array(%c120_i32, %0) : (i32, memref<120x120x120xf64>) -> ()
    }
    return %c0_i32 : i32
  }
  func @init_array(%arg0: i32, %arg1: memref<120x120x120xf64>, %arg2: memref<120x120x120xf64>) {
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
    store %17, %arg2[%2, %5, %9] : memref<120x120x120xf64>
    %18 = load %arg2[%2, %5, %9] : memref<120x120x120xf64>
    store %18, %arg1[%2, %5, %9] : memref<120x120x120xf64>
    %19 = addi %7, %c1_i32 : i32
    br ^bb5(%19 : i32)
  ^bb7:  // pred: ^bb5
    %20 = addi %3, %c1_i32 : i32
    br ^bb3(%20 : i32)
  }
  func @polybench_timer_start() {
    %c0 = constant 0 : index
    call @polybench_prepare_instruments() : () -> ()
    %0 = get_global_memref @polybench_t_start : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          affine.for %arg7 = 1 to #map0()[%0] {
            call @S0(%arg3, %arg5, %arg6, %arg7, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
          }
        }
      }
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          affine.for %arg7 = 1 to #map0()[%0] {
            call @S1(%arg2, %arg5, %arg6, %arg7, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
          }
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
  func @print_array(%arg0: i32, %arg1: memref<120x120x120xf64>) {
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
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb4
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
      %51 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %52 = llvm.load %51 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %53 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
      %54 = llvm.getelementptr %53[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %55 = llvm.call @fprintf(%52, %54) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %42 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %43 = llvm.load %42 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %44 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
    %45 = llvm.getelementptr %44[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %46 = load %arg1[%15, %30, %34] : memref<120x120x120xf64>
    %47 = llvm.mlir.cast %46 : f64 to !llvm.double
    %48 = llvm.call @fprintf(%43, %45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %49 = addi %32, %c1_i32 : i32
    br ^bb5(%49 : i32)
  ^bb7:  // pred: ^bb5
    %50 = addi %28, %c1_i32 : i32
    br ^bb3(%50 : i32)
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
  func private @S0(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.250000e-01 : f64
    %0 = affine.load %arg4[%arg1 + 1, %arg2, %arg3] : memref<120x120x120xf64>
    %1 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %2 = affine.load %arg4[%arg1 - 1, %arg2, %arg3] : memref<120x120x120xf64>
    %3 = affine.load %arg4[%arg1, %arg2 + 1, %arg3] : memref<120x120x120xf64>
    %4 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %5 = affine.load %arg4[%arg1, %arg2 - 1, %arg3] : memref<120x120x120xf64>
    %6 = affine.load %arg4[%arg1, %arg2, %arg3 + 1] : memref<120x120x120xf64>
    %7 = mulf %cst, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst_0, %9 : f64
    %11 = mulf %cst, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst_0, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %17 = mulf %cst, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[%arg1, %arg2, %arg3 - 1] : memref<120x120x120xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst_0, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    return
  }
  func private @S1(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 1.250000e-01 : f64
    %0 = affine.load %arg4[%arg1 + 1, %arg2, %arg3] : memref<120x120x120xf64>
    %1 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %2 = affine.load %arg4[%arg1 - 1, %arg2, %arg3] : memref<120x120x120xf64>
    %3 = affine.load %arg4[%arg1, %arg2 + 1, %arg3] : memref<120x120x120xf64>
    %4 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %5 = affine.load %arg4[%arg1, %arg2 - 1, %arg3] : memref<120x120x120xf64>
    %6 = affine.load %arg4[%arg1, %arg2, %arg3 + 1] : memref<120x120x120xf64>
    %7 = mulf %cst, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst_0, %9 : f64
    %11 = mulf %cst, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst_0, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %17 = mulf %cst, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[%arg1, %arg2, %arg3 - 1] : memref<120x120x120xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst_0, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    return
  }
  func @kernel_heat_3d_new(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
    %c1 = constant 1 : index
    %c500 = constant 500 : index
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = -1 to 63 {
      affine.for %arg5 = #map1(%arg4) to min #map2(%arg4)[%0] {
        affine.if #set0(%arg4, %arg5)[%0] {
          affine.if #set1(%arg4) {
            affine.for %arg6 = #map3(%arg4) to #map4(%arg4)[%0] {
              affine.for %arg7 = max #map5(%arg4, %arg6) to min #map6(%arg4, %arg6)[%0] {
                affine.if #set2(%arg4)[%0] {
                  %1 = affine.apply #map7(%arg4)
                  %2 = affine.apply #map8()[%0]
                  call @S0(%arg3, %1, %2, %c1, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.for %arg6 = max #map9(%arg4, %arg5)[%0] to min #map10(%arg4, %arg5)[%0] {
          affine.if #set3(%arg4, %arg5, %arg6)[%0] {
            affine.if #set4()[%0] {
              affine.for %arg7 = max #map11(%arg4, %arg5, %arg6)[%0] to min #map12(%arg4, %arg5, %arg6)[%0] {
                %1 = affine.apply #map13(%arg6)[%0]
                %2 = affine.apply #map14(%arg6, %arg7)[%0]
                %3 = affine.apply #map8()[%0]
                call @S0(%arg3, %1, %2, %3, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set0(%arg4, %arg5)[%0] {
            affine.if #set1(%arg4) {
              affine.for %arg7 = #map15(%arg6) to min #map6(%arg4, %arg6)[%0] {
                affine.if #set2(%arg4)[%0] {
                  %1 = affine.apply #map7(%arg4)
                  %2 = affine.apply #map8()[%0]
                  %3 = affine.apply #map16(%arg4, %arg7)
                  call @S0(%arg3, %1, %2, %3, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
          affine.for %arg7 = max #map17(%arg4, %arg5, %arg6)[%0] to min #map18(%arg4, %arg5, %arg6)[%0] {
            affine.if #set5(%arg4, %arg5)[%0] {
              affine.if #set6()[%0] {
                affine.for %arg8 = max #map19(%arg5, %arg6)[%0] to min #map20(%arg5, %arg6) {
                  affine.for %arg9 = max #map19(%arg5, %arg7)[%0] to min #map20(%arg5, %arg7) {
                    %1 = affine.apply #map21(%arg5)[%0]
                    %2 = affine.apply #map8()[%0]
                    %3 = affine.apply #map14(%arg5, %arg8)[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set7(%arg4, %arg5, %arg6)[%0] {
              affine.if #set6()[%0] {
                affine.for %arg8 = max #map22(%arg4, %arg5, %arg6)[%0] to min #map12(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map19(%arg6, %arg7)[%0] to min #map20(%arg6, %arg7) {
                    %1 = affine.apply #map21(%arg6)[%0]
                    %2 = affine.apply #map14(%arg6, %arg8)[%0]
                    %3 = affine.apply #map8()[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set8(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set6()[%0] {
                affine.for %arg8 = max #map22(%arg4, %arg5, %arg7)[%0] to min #map12(%arg4, %arg5, %arg7)[%0] {
                  affine.for %arg9 = max #map19(%arg6, %arg7)[%0] to #map23(%arg6) {
                    %1 = affine.apply #map21(%arg7)[%0]
                    %2 = affine.apply #map14(%arg7, %arg8)[%0]
                    %3 = affine.apply #map14(%arg7, %arg9)[%0]
                    call @S1(%arg2, %1, %2, %3, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set9(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map24(%arg4, %arg6) to min #map25(%arg4, %arg6)[%0] {
                affine.for %arg9 = max #map24(%arg4, %arg7) to min #map25(%arg4, %arg7)[%0] {
                  affine.if #set10(%arg4) {
                    %1 = affine.apply #map26(%arg4)
                    %2 = affine.apply #map27(%arg4, %arg8)
                    call @S1(%arg2, %1, %c1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.for %arg8 = max #map28(%arg4, %arg5, %arg6, %arg7)[%0] to min #map29(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set11(%arg4, %arg5, %arg8) {
                affine.for %arg9 = max #map30(%arg6, %arg8) to min #map31(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map30(%arg7, %arg8) to min #map31(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    call @S0(%arg3, %arg8, %c1, %1, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map33(%arg4, %arg5, %arg8) to #map34(%arg4, %arg5, %arg8) {
                affine.for %arg10 = max #map30(%arg6, %arg8) to min #map31(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map30(%arg7, %arg8) to min #map31(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    %2 = affine.apply #map32(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %1, %2, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map35(%arg4, %arg5, %arg8) to min #map36(%arg4, %arg5, %arg8)[%0] {
                affine.if #set12(%arg6, %arg8) {
                  affine.for %arg10 = max #map30(%arg7, %arg8) to min #map31(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    call @S0(%arg3, %arg8, %1, %c1, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
                affine.for %arg10 = max #map37(%arg6, %arg8) to min #map31(%arg6, %arg8)[%0] {
                  affine.if #set12(%arg7, %arg8) {
                    %1 = affine.apply #map32(%arg8, %arg9)
                    %2 = affine.apply #map32(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %1, %2, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                  affine.for %arg11 = max #map37(%arg7, %arg8) to min #map31(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map38(%arg8, %arg9)
                    %2 = affine.apply #map38(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                    %3 = affine.apply #map32(%arg8, %arg9)
                    %4 = affine.apply #map32(%arg8, %arg10)
                    call @S0(%arg3, %arg8, %3, %4, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                  affine.if #set13(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map38(%arg8, %arg9)
                    %2 = affine.apply #map38(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
                affine.if #set13(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map37(%arg7, %arg8) to min #map39(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map38(%arg8, %arg9)
                    %2 = affine.apply #map8()[%0]
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = #map40(%arg4, %arg5, %arg8) to min #map41(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg10 = max #map37(%arg6, %arg8) to min #map39(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map37(%arg7, %arg8) to min #map39(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map38(%arg8, %arg9)
                    %2 = affine.apply #map38(%arg8, %arg10)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.if #set14(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg9 = max #map37(%arg6, %arg8) to min #map39(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map37(%arg7, %arg8) to min #map39(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map8()[%0]
                    %2 = affine.apply #map38(%arg8, %arg9)
                    call @S1(%arg2, %arg8, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set15(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set6()[%0] {
                affine.for %arg8 = max #map42(%arg4, %arg5, %arg6)[%0] to min #map43(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map42(%arg4, %arg5, %arg7)[%0] to min #map43(%arg4, %arg5, %arg7)[%0] {
                    %1 = affine.apply #map44(%arg4, %arg5)[%0]
                    %2 = affine.apply #map8()[%0]
                    %3 = affine.apply #map45(%arg4, %arg5, %arg8)[%0]
                    call @S0(%arg3, %1, %2, %3, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set16(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map46(%arg4, %arg5, %arg6) to min #map47(%arg4, %arg5, %arg6)[%0] {
                affine.for %arg9 = #map15(%arg7) to min #map48(%arg6, %arg7)[%0] {
                  %1 = affine.apply #map49(%arg6)
                  %2 = affine.apply #map50(%arg6, %arg8)
                  call @S0(%arg3, %1, %2, %c1, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
            affine.if #set17(%arg4, %arg5, %arg7)[%0] {
              affine.for %arg8 = max #map46(%arg4, %arg5, %arg7) to min #map47(%arg4, %arg5, %arg7)[%0] {
                affine.for %arg9 = max #map51(%arg6, %arg7) to min #map48(%arg6, %arg7)[%0] {
                  %1 = affine.apply #map49(%arg7)
                  %2 = affine.apply #map50(%arg7, %arg8)
                  %3 = affine.apply #map50(%arg7, %arg9)
                  call @S0(%arg3, %1, %2, %3, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
            affine.if #set18(%arg4, %arg5, %arg6, %arg7) {
              affine.for %arg8 = #map15(%arg6) to min #map52(%arg4, %arg6)[%0] {
                affine.for %arg9 = #map15(%arg7) to min #map52(%arg4, %arg7)[%0] {
                  affine.if #set10(%arg4) {
                    %1 = affine.apply #map53(%arg4)
                    %2 = affine.apply #map54(%arg4, %arg8)
                    call @S0(%arg3, %1, %c1, %2, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
          }
          affine.if #set19(%arg4, %arg6) {
            affine.if #set20()[%0] {
              affine.for %arg7 = max #map46(%arg4, %arg5, %arg6) to min #map55(%arg4, %arg5, %arg6) {
                %1 = affine.apply #map49(%arg6)
                %2 = affine.apply #map50(%arg6, %arg7)
                call @S0(%arg3, %1, %2, %c1, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set21(%arg4, %arg5, %arg6) {
            affine.if #set22(%arg4)[%0] {
              affine.for %arg7 = #map15(%arg6) to #map23(%arg6) {
                affine.if #set10(%arg4) {
                  %1 = affine.apply #map53(%arg4)
                  %2 = affine.apply #map54(%arg4, %arg7)
                  call @S0(%arg3, %1, %c1, %2, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
          affine.if #set23(%arg4, %arg5, %arg6) {
            affine.if #set24(%arg4)[%0] {
              affine.for %arg7 = max #map56(%arg4, %arg6) to #map23(%arg6) {
                %1 = affine.apply #map53(%arg4)
                %2 = affine.apply #map57(%arg4, %arg5)
                %3 = affine.apply #map58(%arg4, %arg7)
                call @S1(%arg2, %1, %2, %3, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set25(%arg4, %arg6) {
            affine.if #set26()[%0] {
              affine.for %arg7 = max #map59(%arg4, %arg5) to min #map60(%arg4, %arg5) {
                affine.for %arg8 = max #map61(%arg6) to #map23(%arg6) {
                  %1 = affine.apply #map62(%arg7)
                  %2 = affine.apply #map62(%arg8)
                  call @S1(%arg2, %c500, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set27(%arg4, %arg5) {
          affine.if #set22(%arg4)[%0] {
            affine.for %arg6 = #map1(%arg4) to #map63(%arg4)[%0] {
              affine.for %arg7 = max #map64(%arg4, %arg6) to min #map52(%arg4, %arg6)[%0] {
                affine.if #set10(%arg4) {
                  %1 = affine.apply #map53(%arg4)
                  %2 = affine.apply #map8()[%0]
                  call @S0(%arg3, %1, %c1, %2, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set28(%arg4, %arg5) {
          affine.if #set24(%arg4)[%0] {
            affine.for %arg6 = #map65(%arg4) to #map66(%arg4)[%0] {
              affine.for %arg7 = max #map56(%arg4, %arg6) to min #map67(%arg4, %arg6)[%0] {
                %1 = affine.apply #map53(%arg4)
                %2 = affine.apply #map57(%arg4, %arg5)
                %3 = affine.apply #map8()[%0]
                call @S1(%arg2, %1, %2, %3, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
        }
        affine.if #set29(%arg4) {
          affine.if #set26()[%0] {
            affine.for %arg6 = 31 to #map68()[%0] {
              affine.for %arg7 = max #map59(%arg4, %arg5) to min #map60(%arg4, %arg5) {
                affine.for %arg8 = max #map61(%arg6) to min #map69(%arg6)[%0] {
                  %1 = affine.apply #map62(%arg7)
                  %2 = affine.apply #map8()[%0]
                  call @S1(%arg2, %c500, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set30(%arg4) {
        affine.if #set31()[%0] {
          affine.for %arg5 = 0 to #map70()[%0] {
            affine.for %arg6 = 0 to #map70()[%0] {
              affine.for %arg7 = max #map71(%arg5) to min #map72(%arg5)[%0] {
                affine.for %arg8 = max #map71(%arg6) to min #map72(%arg6)[%0] {
                  %1 = affine.apply #map8()[%0]
                  %2 = affine.apply #map73(%arg7)
                  call @S0(%arg3, %c1, %1, %2, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set32(%arg4) {
        affine.if #set26()[%0] {
          affine.for %arg5 = 31 to #map68()[%0] {
            affine.for %arg6 = 31 to #map68()[%0] {
              affine.for %arg7 = max #map61(%arg5) to min #map69(%arg5)[%0] {
                affine.for %arg8 = max #map61(%arg6) to min #map69(%arg6)[%0] {
                  %1 = affine.apply #map8()[%0]
                  %2 = affine.apply #map62(%arg7)
                  call @S1(%arg2, %c500, %1, %2, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
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

