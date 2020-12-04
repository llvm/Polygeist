#map0 = affine_map<(d0) -> (d0 ceildiv 2, (d0 * 16 - 498) ceildiv 16)>
#map1 = affine_map<(d0) -> (94, (d0 * 16 + 2013) floordiv 32 + 1, d0 + 64)>
#map2 = affine_map<(d0) -> (d0 * 32 + 1)>
#map3 = affine_map<(d0) -> (d0 * 32 + 3)>
#map4 = affine_map<(d0, d1) -> (d0 * -32 + d1 * 32 + 1997)>
#map5 = affine_map<(d0, d1) -> (d0 * 32 + 32, 2997, d1 * -32 + d0 * 32 + 1999)>
#map6 = affine_map<()[s0, s1] -> ((s0 - 1) floordiv 8 + 1, (s0 * 4 + s1 - 8) floordiv 32 + 1)>
#map7 = affine_map<()[s0] -> (s0 - 1)>
#map8 = affine_map<(d0)[s0] -> (d0 ceildiv 2, (d0 * 16 - s0 + 2) ceildiv 16)>
#map9 = affine_map<(d0)[s0, s1] -> ((s0 * 2 + s1 - 4) floordiv 32 + 1, (d0 * 16 + s1 + 13) floordiv 32 + 1, (d0 * 32 + s1 + 28) floordiv 32 + 1)>
#map10 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 * 32 + s0 * 4 - 3)>
#map11 = affine_map<(d0, d1)[s0, s1] -> (d0 * 32 + 32, s0 * 2 + s1 - 3, d1 * -32 + d0 * 32 + s0 * 4 - 1)>
#set0 = affine_set<(d0) : (d0 * 8 - 499 == 0)>
#set1 = affine_set<() : (3 == 0)>
#set2 = affine_set<(d0) : (d0 + 1 == 0)>
#set3 = affine_set<(d0, d1) : (d1 - d0 - 32 >= 0, d0 - (d1 * 32 - 1030) ceildiv 32 >= 0, d1 - 93 >= 0)>
#set4 = affine_set<() : (13 == 0)>
#set5 = affine_set<(d0) : (d0 - 62 >= 0)>
#set6 = affine_set<() : (21 == 0)>
#set7 = affine_set<() : (1997 == 0)>
#set8 = affine_set<(d0)[s0] : (d0 * 8 - (s0 - 1) == 0)>
#set9 = affine_set<()[s0] : ((s0 + 15) mod 16 == 0)>
#set10 = affine_set<(d0, d1)[s0, s1] : ((d1 * 32 + s0 * 2 - s1 + 1) floordiv 32 - d0 >= 0, d0 - (d1 * 32 + s0 * 2 - s1 - 30) ceildiv 32 >= 0, d1 - (s0 * 2 + s1 - 34) ceildiv 32 >= 0)>
#set11 = affine_set<()[s0] : ((s0 + 29) mod 32 == 0)>
#set12 = affine_set<(d0)[s0] : (d0 - (s0 - 8) ceildiv 8 >= 0)>
#set13 = affine_set<()[s0, s1] : ((s0 * 2 + s1 + 29) mod 32 == 0)>
#set14 = affine_set<()[s0] : (s0 - 3 == 0)>
#set15 = affine_set<()[s0] : ((s0 + 7) mod 8 == 0)>
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
    %c2000_i32 = constant 2000 : i32
    %c0_i32 = constant 0 : i32
    %c2_i32 = constant 2 : i32
    %c3_i32 = constant 3 : i32
    %c1_i32 = constant 1 : i32
    %c499 = constant 499 : index
    %c0 = constant 0 : index
    %0 = alloc() : memref<2000xf64>
    %1 = alloc() : memref<2000xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%2: i32):  // 2 preds: ^bb0, ^bb2
    %3 = cmpi "slt", %2, %c2000_i32 : i32
    %4 = index_cast %2 : i32 to index
    cond_br %3, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = sitofp %2 : i32 to f64
    %6 = sitofp %c2_i32 : i32 to f64
    %7 = addf %5, %6 : f64
    %8 = sitofp %c2000_i32 : i32 to f64
    %9 = divf %7, %8 : f64
    store %9, %0[%4] : memref<2000xf64>
    %10 = sitofp %c3_i32 : i32 to f64
    %11 = addf %5, %10 : f64
    %12 = divf %11, %8 : f64
    store %12, %1[%4] : memref<2000xf64>
    %13 = addi %2, %c1_i32 : i32
    br ^bb1(%13 : i32)
  ^bb3:  // pred: ^bb1
    affine.for %arg2 = -1 to 63 {
      affine.if #set0(%arg2) {
        affine.if #set1() {
          call @S1(%0, %c499, %1) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.for %arg3 = max #map0(%arg2) to min #map1(%arg2) {
        affine.if #set2(%arg2) {
          affine.for %arg4 = #map2(%arg3) to #map3(%arg3) {
            call @S0(%1, %c0, %0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
          }
        }
        affine.for %arg4 = #map4(%arg2, %arg3) to min #map5(%arg2, %arg3) {
          call @S1(%0, %c499, %1) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
        affine.if #set3(%arg2, %arg3) {
          call @S1(%0, %c499, %1) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.if #set2(%arg2) {
        affine.if #set4() {
          call @S0(%1, %c0, %0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.if #set5(%arg2) {
        affine.if #set6() {
          call @S1(%0, %c499, %1) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    affine.if #set7() {
      affine.if #set1() {
        affine.if #set1() {
          call @S1(%0, %c499, %1) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    call @print_array(%c2000_i32, %0) : (i32, memref<2000xf64>) -> ()
    return %c0_i32 : i32
  }
  func private @print_array(%arg0: i32, %arg1: memref<2000xf64>) {
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
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb2
    %14 = cmpi "slt", %13, %arg0 : i32
    %15 = index_cast %13 : i32 to index
    cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = remi_signed %13, %c20_i32 : i32
    %17 = cmpi "eq", %16, %c0_i32 : i32
    scf.if %17 {
      %38 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %39 = llvm.load %38 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %40 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
      %41 = llvm.getelementptr %40[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %42 = llvm.call @fprintf(%39, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %18 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %19 = llvm.load %18 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %20 = llvm.mlir.addressof @str4 : !llvm.ptr<array<8 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %22 = load %arg1[%15] : memref<2000xf64>
    %23 = llvm.mlir.cast %22 : f64 to !llvm.double
    %24 = llvm.call @fprintf(%19, %21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %25 = addi %13, %c1_i32 : i32
    br ^bb1(%25 : i32)
  ^bb3:  // pred: ^bb1
    %26 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %27 = llvm.load %26 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %28 = llvm.mlir.addressof @str5 : !llvm.ptr<array<17 x i8>>
    %29 = llvm.getelementptr %28[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %30 = llvm.mlir.addressof @str2 : !llvm.ptr<array<2 x i8>>
    %31 = llvm.getelementptr %30[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %32 = llvm.call @fprintf(%27, %29, %31) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %35 = llvm.mlir.addressof @str6 : !llvm.ptr<array<23 x i8>>
    %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %37 = llvm.call @fprintf(%34, %36) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.load %arg2[%arg1 - 1] : memref<2000xf64>
    %1 = affine.load %arg2[%arg1] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg2[%arg1 + 1] : memref<2000xf64>
    %4 = addf %2, %3 : f64
    %5 = mulf %cst, %4 : f64
    affine.store %5, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func private @S1(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.load %arg2[%arg1 - 1] : memref<2000xf64>
    %1 = affine.load %arg2[%arg1] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg2[%arg1 + 1] : memref<2000xf64>
    %4 = addf %2, %3 : f64
    %5 = mulf %cst, %4 : f64
    affine.store %5, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @kernel_jacobi_1d_new(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %c0 = constant 0 : index
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg4 = -1 to min #map6()[%1, %0] {
      affine.if #set8(%arg4)[%1] {
        affine.if #set9()[%1] {
          %2 = affine.apply #map7()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.for %arg5 = max #map8(%arg4)[%1] to min #map9(%arg4)[%1, %0] {
        affine.if #set2(%arg4) {
          affine.for %arg6 = #map2(%arg5) to #map3(%arg5) {
            call @S0(%arg3, %c0, %arg2) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
          }
        }
        affine.for %arg6 = #map10(%arg4, %arg5)[%1] to min #map11(%arg4, %arg5)[%1, %0] {
          %2 = affine.apply #map7()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
        affine.if #set10(%arg4, %arg5)[%1, %0] {
          %2 = affine.apply #map7()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.if #set2(%arg4) {
        affine.if #set11()[%0] {
          call @S0(%arg3, %c0, %arg2) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.if #set12(%arg4)[%1] {
        affine.if #set13()[%1, %0] {
          %2 = affine.apply #map7()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    affine.if #set14()[%0] {
      affine.if #set15()[%1] {
        affine.if #set9()[%1] {
          %2 = affine.apply #map7()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    return
  }
}

