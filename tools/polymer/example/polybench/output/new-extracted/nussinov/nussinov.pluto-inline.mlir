#map0 = affine_map<(d0)[s0] -> (d0 * -32 + s0 - 31)>
#map1 = affine_map<(d0) -> (d0 * 32 + 31)>
#map2 = affine_map<(d0) -> (0, (d0 * 32 - 2499) ceildiv 32)>
#map3 = affine_map<(d0) -> (79, d0 + 1)>
#map4 = affine_map<(d0, d1) -> (2, d0 * 32 - d1 * 32, d1 * -32 + 2470)>
#map5 = affine_map<(d0, d1) -> (2500, d0 * 32 - d1 * 32 + 32)>
#map6 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map7 = affine_map<(d0, d1) -> (d0 * 32, -d1 + 2501)>
#map8 = affine_map<(d0) -> (2500, d0 * 32 + 32)>
#map9 = affine_map<(d0) -> (-d0 + 2500)>
#map10 = affine_map<(d0) -> (d0)>
#map11 = affine_map<()[s0] -> ((s0 - 62) floordiv 32 + 1)>
#map12 = affine_map<()[s0] -> (0, (s0 - 61) ceildiv 32)>
#map13 = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>
#map14 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map15 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map16 = affine_map<()[s0] -> (s0 - 1)>
#map17 = affine_map<(d0, d1)[s0] -> (2, d0 * 32 - d1 * 32, d1 * -32 + s0 - 30)>
#map18 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map19 = affine_map<(d0, d1)[s0] -> (d0 * 32, -d1 + s0 + 1)>
#map20 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#set0 = affine_set<() : (2438 >= 0)>
#set1 = affine_set<() : (6 == 0)>
#set2 = affine_set<(d0, d1) : (-d0 + 77 >= 0, -d1 + 77 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 - d1 == 0, d0 - 78 >= 0)>
#set4 = affine_set<(d0, d1) : ((-d1 + 2500) floordiv 32 - d0 >= 0)>
#set5 = affine_set<()[s0] : (s0 - 62 >= 0)>
#set6 = affine_set<()[s0] : ((s0 + 2) mod 32 == 0)>
#set7 = affine_set<(d0, d1)[s0] : (-d0 + (s0 - 31) floordiv 32 >= 0, -d1 + s0 floordiv 32 - 1 >= 0)>
#set8 = affine_set<(d0, d1)[s0] : (d0 - d1 == 0, d0 - (s0 - 31) ceildiv 32 >= 0)>
#set9 = affine_set<(d0, d1)[s0] : ((-d1 + s0) floordiv 32 - d0 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%d \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("table\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c2500_i32 = constant 2500 : i32
    %c0_i32 = constant 0 : i32
    %c4_i32 = constant 4 : i32
    %c1_i32 = constant 1 : i32
    %c2499 = constant 2499 : index
    %c2500 = constant 2500 : index
    %c1 = constant 1 : index
    %0 = alloc() : memref<2500xi8>
    %1 = alloc() : memref<2500x2500xi32>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%2: i32):  // 2 preds: ^bb0, ^bb2
    %3 = cmpi "slt", %2, %c2500_i32 : i32
    %4 = index_cast %2 : i32 to index
    cond_br %3, ^bb2, ^bb3(%c0_i32 : i32)
  ^bb2:  // pred: ^bb1
    %5 = addi %2, %c1_i32 : i32
    %6 = remi_signed %5, %c4_i32 : i32
    %7 = trunci %6 : i32 to i8
    store %7, %0[%4] : memref<2500xi8>
    br ^bb1(%5 : i32)
  ^bb3(%8: i32):  // 2 preds: ^bb1, ^bb7
    %9 = cmpi "slt", %8, %c2500_i32 : i32
    %10 = index_cast %8 : i32 to index
    cond_br %9, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    affine.if #set0() {
      affine.if #set1() {
        affine.for %arg2 = 0 to 77 {
          %16 = affine.apply #map0(%arg2)[%c2500]
          %17 = affine.apply #map1(%arg2)
          call @S0(%1, %16, %17, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          %18 = affine.apply #map0(%arg2)[%c2500]
          %19 = affine.apply #map1(%arg2)
          call @S1(%1, %18, %19, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          %20 = affine.apply #map0(%arg2)[%c2500]
          %21 = affine.apply #map1(%arg2)
          call @S2(%1, %20, %21, %c2500, %0) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
          %22 = affine.apply #map0(%arg2)[%c2500]
          %23 = affine.apply #map1(%arg2)
          call @S3(%1, %22, %23, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
        }
      }
    }
    affine.for %arg2 = 77 to 157 {
      affine.for %arg3 = max #map2(%arg2) to min #map3(%arg2) {
        affine.if #set2(%arg2, %arg3) {
          %16 = affine.apply #map0(%arg3)[%c2500]
          %17 = affine.apply #map1(%arg3)
          call @S0(%1, %16, %17, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          %18 = affine.apply #map0(%arg3)[%c2500]
          %19 = affine.apply #map1(%arg3)
          call @S1(%1, %18, %19, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          %20 = affine.apply #map0(%arg3)[%c2500]
          %21 = affine.apply #map1(%arg3)
          call @S2(%1, %20, %21, %c2500, %0) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
          %22 = affine.apply #map0(%arg3)[%c2500]
          %23 = affine.apply #map1(%arg3)
          call @S3(%1, %22, %23, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
        }
        affine.if #set3(%arg2, %arg3) {
          call @S0(%1, %c1, %c2499, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          call @S1(%1, %c1, %c2499, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          call @S2(%1, %c1, %c2499, %c2500, %0) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
          call @S3(%1, %c1, %c2499, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
        }
        affine.for %arg4 = max #map4(%arg2, %arg3) to min #map5(%arg2, %arg3) {
          affine.if #set4(%arg3, %arg4) {
            %16 = affine.apply #map6(%arg4)[%c2500]
            call @S0(%1, %arg4, %16, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
            %17 = affine.apply #map6(%arg4)[%c2500]
            call @S1(%1, %arg4, %17, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
            %18 = affine.apply #map6(%arg4)[%c2500]
            call @S2(%1, %arg4, %18, %c2500, %0) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
            %19 = affine.apply #map6(%arg4)[%c2500]
            call @S3(%1, %arg4, %19, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
          }
          affine.for %arg5 = max #map7(%arg3, %arg4) to min #map8(%arg3) {
            call @S0(%1, %arg4, %arg5, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
            call @S1(%1, %arg4, %arg5, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
            call @S2(%1, %arg4, %arg5, %c2500, %0) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
            call @S3(%1, %arg4, %arg5, %c2500) : (memref<2500x2500xi32>, index, index, index) -> ()
            affine.for %arg6 = #map9(%arg4) to #map10(%arg5) {
              call @S4(%1, %arg4, %arg5, %c2500, %arg6) : (memref<2500x2500xi32>, index, index, index, index) -> ()
            }
          }
        }
      }
    }
    call @print_array(%c2500_i32, %1) : (i32, memref<2500x2500xi32>) -> ()
    return %c0_i32 : i32
  ^bb5(%11: i32):  // 2 preds: ^bb3, ^bb6
    %12 = cmpi "slt", %11, %c2500_i32 : i32
    %13 = index_cast %11 : i32 to index
    cond_br %12, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    store %c0_i32, %1[%10, %13] : memref<2500x2500xi32>
    %14 = addi %11, %c1_i32 : i32
    br ^bb5(%14 : i32)
  ^bb7:  // pred: ^bb5
    %15 = addi %8, %c1_i32 : i32
    br ^bb3(%15 : i32)
  }
  func private @print_array(%arg0: i32, %arg1: memref<2500x2500xi32>) {
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
    %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<6 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    br ^bb1(%c0_i32, %c0_i32 : i32, i32)
  ^bb1(%13: i32, %14: i32):  // 2 preds: ^bb0, ^bb5
    %15 = cmpi "slt", %13, %arg0 : i32
    %16 = index_cast %13 : i32 to index
    cond_br %15, ^bb3(%13, %14 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    %17 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %18 = llvm.load %17 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %19 = llvm.mlir.addressof @str5 : !llvm.ptr<array<17 x i8>>
    %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %21 = llvm.mlir.addressof @str2 : !llvm.ptr<array<6 x i8>>
    %22 = llvm.getelementptr %21[%3, %3] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %23 = llvm.call @fprintf(%18, %20, %22) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %24 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %25 = llvm.load %24 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %26 = llvm.mlir.addressof @str6 : !llvm.ptr<array<23 x i8>>
    %27 = llvm.getelementptr %26[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %28 = llvm.call @fprintf(%25, %27) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  ^bb3(%29: i32, %30: i32):  // 2 preds: ^bb1, ^bb4
    %31 = cmpi "slt", %29, %arg0 : i32
    %32 = index_cast %29 : i32 to index
    cond_br %31, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %33 = remi_signed %30, %c20_i32 : i32
    %34 = cmpi "eq", %33, %c0_i32 : i32
    scf.if %34 {
      %45 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %46 = llvm.load %45 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %47 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
      %48 = llvm.getelementptr %47[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %49 = llvm.call @fprintf(%46, %48) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str4 : !llvm.ptr<array<4 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg1[%16, %32] : memref<2500x2500xi32>
    %40 = llvm.mlir.cast %39 : i32 to !llvm.i32
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
    %42 = addi %30, %c1_i32 : i32
    %43 = addi %29, %c1_i32 : i32
    br ^bb3(%43, %42 : i32, i32)
  ^bb5:  // pred: ^bb3
    %44 = addi %13, %c1_i32 : i32
    br ^bb1(%44, %30 : i32, i32)
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<2500x2500xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    %1 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2 - 1] : memref<2500x2500xi32>
    %2 = cmpi "sge", %0, %1 : i32
    %3 = scf.if %2 -> (i32) {
      %4 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
      scf.yield %4 : i32
    } else {
      %4 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2 - 1] : memref<2500x2500xi32>
      scf.yield %4 : i32
    }
    affine.store %3, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    return
  }
  func private @S1(%arg0: memref<2500x2500xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    %1 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2] : memref<2500x2500xi32>
    %2 = cmpi "sge", %0, %1 : i32
    %3 = scf.if %2 -> (i32) {
      %4 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
      scf.yield %4 : i32
    } else {
      %4 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2] : memref<2500x2500xi32>
      scf.yield %4 : i32
    }
    affine.store %3, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    return
  }
  func private @S2(%arg0: memref<2500x2500xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<2500xi8>) attributes {scop.stmt} {
    %c3_i32 = constant 3 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    %1 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2 - 1] : memref<2500x2500xi32>
    %2 = affine.load %arg4[-%arg1 + symbol(%arg3) - 1] : memref<2500xi8>
    %3 = sexti %2 : i8 to i32
    %4 = affine.load %arg4[%arg2] : memref<2500xi8>
    %5 = sexti %4 : i8 to i32
    %6 = addi %3, %5 : i32
    %7 = cmpi "eq", %6, %c3_i32 : i32
    %8 = select %7, %c1_i32, %c0_i32 : i32
    %9 = addi %1, %8 : i32
    %10 = cmpi "sge", %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %12 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
      scf.yield %12 : i32
    } else {
      %12 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2 - 1] : memref<2500x2500xi32>
      %13 = affine.load %arg4[-%arg1 + symbol(%arg3) - 1] : memref<2500xi8>
      %14 = sexti %13 : i8 to i32
      %15 = affine.load %arg4[%arg2] : memref<2500xi8>
      %16 = sexti %15 : i8 to i32
      %17 = addi %14, %16 : i32
      %18 = cmpi "eq", %17, %c3_i32 : i32
      %19 = select %18, %c1_i32, %c0_i32 : i32
      %20 = addi %12, %19 : i32
      scf.yield %20 : i32
    }
    affine.store %11, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    return
  }
  func private @S3(%arg0: memref<2500x2500xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    %1 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2 - 1] : memref<2500x2500xi32>
    %2 = cmpi "sge", %0, %1 : i32
    %3 = scf.if %2 -> (i32) {
      %4 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
      scf.yield %4 : i32
    } else {
      %4 = affine.load %arg0[-%arg1 + symbol(%arg3), %arg2 - 1] : memref<2500x2500xi32>
      scf.yield %4 : i32
    }
    affine.store %3, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    return
  }
  func private @S4(%arg0: memref<2500x2500xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    %1 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg4] : memref<2500x2500xi32>
    %2 = affine.load %arg0[%arg4 + 1, %arg2] : memref<2500x2500xi32>
    %3 = addi %1, %2 : i32
    %4 = cmpi "sge", %0, %3 : i32
    %5 = scf.if %4 -> (i32) {
      %6 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
      scf.yield %6 : i32
    } else {
      %6 = affine.load %arg0[-%arg1 + symbol(%arg3) - 1, %arg4] : memref<2500x2500xi32>
      %7 = affine.load %arg0[%arg4 + 1, %arg2] : memref<2500x2500xi32>
      %8 = addi %6, %7 : i32
      scf.yield %8 : i32
    }
    affine.store %5, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<2500x2500xi32>
    return
  }
  func @kernel_nussinov_new(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    affine.if #set5()[%0] {
      affine.if #set6()[%0] {
        affine.for %arg3 = 0 to #map11()[%0] {
          %1 = affine.apply #map0(%arg3)[%0]
          %2 = affine.apply #map1(%arg3)
          call @S0(%arg2, %1, %2, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          %3 = affine.apply #map0(%arg3)[%0]
          %4 = affine.apply #map1(%arg3)
          call @S1(%arg2, %3, %4, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          %5 = affine.apply #map0(%arg3)[%0]
          %6 = affine.apply #map1(%arg3)
          call @S2(%arg2, %5, %6, %0, %arg1) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
          %7 = affine.apply #map0(%arg3)[%0]
          %8 = affine.apply #map1(%arg3)
          call @S3(%arg2, %7, %8, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
        }
      }
    }
    affine.for %arg3 = max #map12()[%0] to #map13()[%0] {
      affine.for %arg4 = max #map14(%arg3)[%0] to min #map15(%arg3)[%0] {
        affine.if #set7(%arg3, %arg4)[%0] {
          %1 = affine.apply #map0(%arg4)[%0]
          %2 = affine.apply #map1(%arg4)
          call @S0(%arg2, %1, %2, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          %3 = affine.apply #map0(%arg4)[%0]
          %4 = affine.apply #map1(%arg4)
          call @S1(%arg2, %3, %4, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          %5 = affine.apply #map0(%arg4)[%0]
          %6 = affine.apply #map1(%arg4)
          call @S2(%arg2, %5, %6, %0, %arg1) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
          %7 = affine.apply #map0(%arg4)[%0]
          %8 = affine.apply #map1(%arg4)
          call @S3(%arg2, %7, %8, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
        }
        affine.if #set8(%arg3, %arg4)[%0] {
          %1 = affine.apply #map16()[%0]
          call @S0(%arg2, %c1, %1, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          %2 = affine.apply #map16()[%0]
          call @S1(%arg2, %c1, %2, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          %3 = affine.apply #map16()[%0]
          call @S2(%arg2, %c1, %3, %0, %arg1) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
          %4 = affine.apply #map16()[%0]
          call @S3(%arg2, %c1, %4, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
        }
        affine.for %arg5 = max #map17(%arg3, %arg4)[%0] to min #map18(%arg3, %arg4)[%0] {
          affine.if #set9(%arg4, %arg5)[%0] {
            %1 = affine.apply #map6(%arg5)[%0]
            call @S0(%arg2, %arg5, %1, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
            %2 = affine.apply #map6(%arg5)[%0]
            call @S1(%arg2, %arg5, %2, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
            %3 = affine.apply #map6(%arg5)[%0]
            call @S2(%arg2, %arg5, %3, %0, %arg1) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
            %4 = affine.apply #map6(%arg5)[%0]
            call @S3(%arg2, %arg5, %4, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
          }
          affine.for %arg6 = max #map19(%arg4, %arg5)[%0] to min #map20(%arg4)[%0] {
            call @S0(%arg2, %arg5, %arg6, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
            call @S1(%arg2, %arg5, %arg6, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
            call @S2(%arg2, %arg5, %arg6, %0, %arg1) : (memref<2500x2500xi32>, index, index, index, memref<2500xi8>) -> ()
            call @S3(%arg2, %arg5, %arg6, %0) : (memref<2500x2500xi32>, index, index, index) -> ()
            affine.for %arg7 = #map6(%arg5)[%0] to #map10(%arg6) {
              call @S4(%arg2, %arg5, %arg6, %0, %arg7) : (memref<2500x2500xi32>, index, index, index, index) -> ()
            }
          }
        }
      }
    }
    return
  }
}

