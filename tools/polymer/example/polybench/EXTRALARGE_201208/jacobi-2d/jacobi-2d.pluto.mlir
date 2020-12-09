#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<()[s0] -> ((-s0 - 29) ceildiv 32)>
#map2 = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>
#map3 = affine_map<(d0) -> (d0, -d0 - 1)>
#map4 = affine_map<(d0)[s0, s1] -> ((d0 * -8 + s0 - 1) floordiv 8 + 1, (d0 * 16 + s1 + 13) floordiv 16 + 1, (s0 * 2 + s1 - 3) floordiv 32 + 1)>
#map5 = affine_map<(d0, d1)[s0] -> (0, (d0 + d1 - 1) ceildiv 2, (d1 * 32 - s0 - 28) ceildiv 32)>
#map6 = affine_map<(d0, d1)[s0, s1] -> ((s0 * 2 + s1 - 3) floordiv 32 + 1, (d0 * 8 + d1 * 24 + s1 + 28) floordiv 32 + 1, (d0 * 16 + d1 * 16 + s1 + 29) floordiv 32 + 1)>
#map7 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 - s0 + 3)>
#map8 = affine_map<(d0, d1) -> (d0 * 32 + 1, d1 * 32 + 32)>
#map9 = affine_map<()[s0] -> (s0 - 2)>
#map10 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 + s0 - 2)>
#map11 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - s0 + 3, d2 * -32 + d1 * 64 - s0 * 2 - 27)>
#map12 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d2 * 64 - s0 * 2 + 5)>
#map13 = affine_map<(d0, d1) -> (d0 * 32, d1 * 32 + 2)>
#map14 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 32 + s0)>
#map15 = affine_map<(d0, d1) -> (d0 * -32 + d1 - 1)>
#map16 = affine_map<(d0, d1, d2)[s0] -> (0, (d0 * 32 - s0 + 2) ceildiv 2, (d1 * 32 - s0 + 2) ceildiv 2, d2 * 8 + d0 * 8, d2 * 16 + 1)>
#map17 = affine_map<(d0, d1, d2)[s0, s1] -> (d0 * 16 + s0 floordiv 2 + 15, s1, d1 * 16 + 15, d2 * 16 + 15, d0 * 8 + d1 * 8 + 16)>
#map18 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 1)>
#map19 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0 - 1)>
#map20 = affine_map<(d0, d1) -> (d0 * -2 + d1)>
#map21 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d1 * 4 - 31)>
#map22 = affine_map<(d0, d1) -> (d0 * -32 + d1 * 4 - 29)>
#map23 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d1 * 4 - 29)>
#map24 = affine_map<(d0, d1, d2)[s0] -> (d0 * -32 + d1 * 4 + 1, d2 * 32 + 32, d1 * 2 + s0 - 1)>
#map25 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
#map26 = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
#map27 = affine_map<(d0, d1) -> (d0 * -32 + d1 * 4 + 1)>
#map28 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d2 * 4 + 3, d2 * 2 + s0)>
#map29 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0)>
#map30 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 + s0 + 30)>
#map31 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 * 2 + 28)>
#map32 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 - s0 - 29)>
#map33 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 + 31, d2 * -32 + d1 * 64 + 29)>
#map34 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d2 * 64 + 61, d2 * 32 + s0 + 29)>
#map35 = affine_map<(d0, d1) -> (d0 * -32 + d1 - 30)>
#map36 = affine_map<(d0) -> (d0 * 32)>
#map37 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29)>
#set0 = affine_set<(d0, d1)[s0] : ((d1 * 16 - s0 + 1) floordiv 16 - d0 >= 0, d1 - (s0 - 1) ceildiv 32 >= 0)>
#set1 = affine_set<()[s0] : ((s0 + 1) mod 2 == 0)>
#set2 = affine_set<(d0, d1, d2)[s0] : ((d1 * -16 + d2 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d2 - d1 - 1 >= 0, d2 - (s0 - 1) ceildiv 32 >= 0)>
#set3 = affine_set<(d0, d1, d2)[s0] : (d0 - d1 == 0, d0 - (d2 * 32 - s0 + 1) ceildiv 32 >= 0)>
#set4 = affine_set<(d0, d1, d2) : (d0 - (d1 - 16) ceildiv 16 >= 0, d1 floordiv 16 - d2 >= 0)>
#set5 = affine_set<(d0, d1) : (d1 floordiv 16 - d0 >= 0)>
#set6 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set7 = affine_set<(d0, d1, d2)[s0] : ((d1 * 2 - s0 + 1) floordiv 32 - d0 >= 0, d2 - (d1 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set8 = affine_set<(d0, d1, d2)[s0, s1] : ((d1 * 16 - s0 + 1) floordiv 16 - d0 >= 0, -d0 + (s1 * 2 - s0 - 31) floordiv 32 >= 0, (d2 * 32 - s0 - 1) floordiv 32 - d0 >= 0)>
#set9 = affine_set<(d0, d1, d2)[s0, s1] : (d0 - (d1 * 32 - s0 + 1) ceildiv 32 >= 0, d0 - (-d2 + d1 * 2) >= 0, -d1 + s1 floordiv 16 - 1 >= 0)>
#set10 = affine_set<(d0, d1, d2)[s0] : (d0 - d1 == 0, -d0 + s0 floordiv 16 - 1 >= 0, d2 - d0 - 1 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("A\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c2800_i32 = constant 2800 : i32
    %c1000_i32 = constant 1000 : i32
    %c42_i32 = constant 42 : i32
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<2800x2800xf64>
    %1 = alloc() : memref<2800x2800xf64>
    call @init_array(%c2800_i32, %0, %1) : (i32, memref<2800x2800xf64>, memref<2800x2800xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    call @kernel_jacobi_2d_new(%c1000_i32, %c2800_i32, %0, %1) : (i32, i32, memref<2800x2800xf64>, memref<2800x2800xf64>) -> ()
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
      call @print_array(%c2800_i32, %0) : (i32, memref<2800x2800xf64>) -> ()
    }
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: memref<2800x2800xf64>, %arg2: memref<2800x2800xf64>) {
    %c0_i32 = constant 0 : i32
    %c2_i32 = constant 2 : i32
    %c3_i32 = constant 3 : i32
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
    %6 = sitofp %0 : i32 to f64
    %7 = addi %3, %c2_i32 : i32
    %8 = sitofp %7 : i32 to f64
    %9 = mulf %6, %8 : f64
    %10 = sitofp %c2_i32 : i32 to f64
    %11 = addf %9, %10 : f64
    %12 = sitofp %arg0 : i32 to f64
    %13 = divf %11, %12 : f64
    store %13, %arg1[%2, %5] : memref<2800x2800xf64>
    %14 = addi %3, %c3_i32 : i32
    %15 = sitofp %14 : i32 to f64
    %16 = mulf %6, %15 : f64
    %17 = sitofp %c3_i32 : i32 to f64
    %18 = addf %16, %17 : f64
    %19 = divf %18, %12 : f64
    store %19, %arg2[%2, %5] : memref<2800x2800xf64>
    %20 = addi %3, %c1_i32 : i32
    br ^bb3(%20 : i32)
  ^bb5:  // pred: ^bb3
    %21 = addi %0, %c1_i32 : i32
    br ^bb1(%21 : i32)
  }
  func private @polybench_timer_start()
  func private @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<2800x2800xf64>, %arg3: memref<2800x2800xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %1 {
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          call @S0(%arg3, %arg5, %arg6, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
        }
      }
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          call @S1(%arg2, %arg5, %arg6, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
        }
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @print_array(%arg0: i32, %arg1: memref<2800x2800xf64>) {
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.mlir.constant(0 : index) : !llvm.i64
    %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
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
    %18 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %19 = llvm.getelementptr %18[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %20 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = llvm.call @fprintf(%17, %19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %23 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %24 = llvm.load %23 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %25 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
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
      %46 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
      %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg1[%15, %30] : memref<2800x2800xf64>
    %40 = llvm.mlir.cast %39 : f64 to !llvm.double
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %42 = addi %28, %c1_i32 : i32
    br ^bb3(%42 : i32)
  ^bb5:  // pred: ^bb3
    %43 = addi %13, %c1_i32 : i32
    br ^bb1(%43 : i32)
  }
  func private @S0(%arg0: memref<2800x2800xf64>, %arg1: index, %arg2: index, %arg3: memref<2800x2800xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[%arg1, %arg2] : memref<2800x2800xf64>
    %1 = affine.load %arg3[%arg1, %arg2 - 1] : memref<2800x2800xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg3[%arg1, %arg2 + 1] : memref<2800x2800xf64>
    %4 = addf %2, %3 : f64
    %5 = affine.load %arg3[%arg1 + 1, %arg2] : memref<2800x2800xf64>
    %6 = addf %4, %5 : f64
    %7 = affine.load %arg3[%arg1 - 1, %arg2] : memref<2800x2800xf64>
    %8 = addf %6, %7 : f64
    %9 = mulf %cst, %8 : f64
    affine.store %9, %arg0[%arg1, %arg2] : memref<2800x2800xf64>
    return
  }
  func private @S1(%arg0: memref<2800x2800xf64>, %arg1: index, %arg2: index, %arg3: memref<2800x2800xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[%arg1, %arg2] : memref<2800x2800xf64>
    %1 = affine.load %arg3[%arg1, %arg2 - 1] : memref<2800x2800xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg3[%arg1, %arg2 + 1] : memref<2800x2800xf64>
    %4 = addf %2, %3 : f64
    %5 = affine.load %arg3[%arg1 + 1, %arg2] : memref<2800x2800xf64>
    %6 = addf %4, %5 : f64
    %7 = affine.load %arg3[%arg1 - 1, %arg2] : memref<2800x2800xf64>
    %8 = addf %6, %7 : f64
    %9 = mulf %cst, %8 : f64
    affine.store %9, %arg0[%arg1, %arg2] : memref<2800x2800xf64>
    return
  }
  func private @kernel_jacobi_2d_new(%arg0: i32, %arg1: i32, %arg2: memref<2800x2800xf64>, %arg3: memref<2800x2800xf64>) {
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg4 = #map1()[%1] to #map2()[%0] {
      affine.for %arg5 = max #map3(%arg4) to min #map4(%arg4)[%0, %1] {
        affine.for %arg6 = max #map5(%arg4, %arg5)[%1] to min #map6(%arg4, %arg5)[%0, %1] {
          affine.if #set0(%arg4, %arg5)[%1] {
            affine.if #set1()[%1] {
              affine.for %arg7 = max #map7(%arg6, %arg5)[%1] to min #map8(%arg5, %arg6) {
                %2 = affine.apply #map9()[%1]
                %3 = affine.apply #map10(%arg5, %arg7)[%1]
                call @S1(%arg2, %2, %3, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
          }
          affine.if #set2(%arg4, %arg5, %arg6)[%1] {
            affine.if #set1()[%1] {
              affine.for %arg7 = max #map11(%arg5, %arg6, %arg4)[%1] to min #map12(%arg5, %arg4, %arg6)[%1] {
                %2 = affine.apply #map10(%arg6, %arg7)[%1]
                %3 = affine.apply #map9()[%1]
                call @S1(%arg2, %2, %3, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
          }
          affine.if #set3(%arg4, %arg5, %arg6)[%1] {
            affine.for %arg7 = max #map13(%arg6, %arg4) to min #map14(%arg6, %arg4)[%1] {
              %2 = affine.apply #map15(%arg4, %arg7)
              call @S1(%arg2, %c1, %2, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
            }
          }
          affine.for %arg7 = max #map16(%arg5, %arg6, %arg4)[%1] to min #map17(%arg4, %arg5, %arg6)[%1, %0] {
            affine.if #set4(%arg4, %arg7, %arg5) {
              affine.for %arg8 = max #map18(%arg6, %arg7) to min #map19(%arg6, %arg7)[%1] {
                %2 = affine.apply #map20(%arg7, %arg8)
                call @S0(%arg3, %c1, %2, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
            affine.for %arg8 = max #map21(%arg5, %arg7, %arg4) to #map22(%arg4, %arg7) {
              affine.for %arg9 = max #map18(%arg6, %arg7) to min #map19(%arg6, %arg7)[%1] {
                %2 = affine.apply #map20(%arg7, %arg8)
                %3 = affine.apply #map20(%arg7, %arg9)
                call @S0(%arg3, %2, %3, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
            affine.for %arg8 = max #map23(%arg5, %arg7, %arg4) to min #map24(%arg4, %arg7, %arg5)[%1] {
              affine.if #set5(%arg6, %arg7) {
                %2 = affine.apply #map20(%arg7, %arg8)
                call @S0(%arg3, %2, %c1, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
              affine.for %arg9 = max #map25(%arg6, %arg7) to min #map19(%arg6, %arg7)[%1] {
                %2 = affine.apply #map20(%arg7, %arg8)
                %3 = affine.apply #map20(%arg7, %arg9)
                call @S0(%arg3, %2, %3, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
                %4 = affine.apply #map26(%arg7, %arg8)
                %5 = affine.apply #map26(%arg7, %arg9)
                call @S1(%arg2, %4, %5, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
              affine.if #set6(%arg6, %arg7)[%1] {
                %2 = affine.apply #map26(%arg7, %arg8)
                %3 = affine.apply #map9()[%1]
                call @S1(%arg2, %2, %3, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
            affine.for %arg8 = #map27(%arg4, %arg7) to min #map28(%arg5, %arg4, %arg7)[%1] {
              affine.for %arg9 = max #map25(%arg6, %arg7) to min #map29(%arg6, %arg7)[%1] {
                %2 = affine.apply #map26(%arg7, %arg8)
                %3 = affine.apply #map26(%arg7, %arg9)
                call @S1(%arg2, %2, %3, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
            affine.if #set7(%arg4, %arg7, %arg5)[%1] {
              affine.for %arg8 = max #map25(%arg6, %arg7) to min #map29(%arg6, %arg7)[%1] {
                %2 = affine.apply #map9()[%1]
                %3 = affine.apply #map26(%arg7, %arg8)
                call @S1(%arg2, %2, %3, %arg3) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
          }
          affine.if #set8(%arg4, %arg5, %arg6)[%1, %0] {
            affine.if #set1()[%1] {
              affine.for %arg7 = max #map30(%arg6, %arg4)[%1] to min #map31(%arg6, %arg4)[%1] {
                %2 = affine.apply #map9()[%1]
                %3 = affine.apply #map32(%arg4, %arg7)[%1]
                call @S0(%arg3, %2, %3, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
              }
            }
          }
          affine.if #set9(%arg4, %arg6, %arg5)[%1, %0] {
            affine.for %arg7 = max #map33(%arg5, %arg6, %arg4) to min #map34(%arg5, %arg4, %arg6)[%1] {
              %2 = affine.apply #map35(%arg6, %arg7)
              call @S0(%arg3, %2, %c1, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
            }
          }
          affine.if #set10(%arg4, %arg5, %arg6)[%0] {
            affine.for %arg7 = #map36(%arg6) to min #map37(%arg6, %arg4)[%1] {
              %2 = affine.apply #map35(%arg4, %arg7)
              call @S0(%arg3, %c1, %2, %arg2) : (memref<2800x2800xf64>, index, index, memref<2800x2800xf64>) -> ()
            }
          }
        }
      }
    }
    return
  }
}

