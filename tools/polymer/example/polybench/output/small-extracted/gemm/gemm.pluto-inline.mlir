#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (20, d0 * 32 + 32)>
#map2 = affine_map<(d0) -> (30, d0 * 32 + 32)>
#map3 = affine_map<(d0) -> (25, d0 * 32 + 32)>
#map4 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map5 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2lf \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("C\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c20_i32 = constant 20 : i32
    %c25_i32 = constant 25 : i32
    %c30_i32 = constant 30 : i32
    %c0 = constant 0 : index
    %cst = constant 1.500000e+00 : f64
    %cst_0 = constant 1.200000e+00 : f64
    %c0_i32 = constant 0 : i32
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloc() : memref<20x25xf64>
    %3 = alloc() : memref<20x30xf64>
    %4 = alloc() : memref<30x25xf64>
    store %cst, %0[%c0] : memref<1xf64>
    store %cst_0, %1[%c0] : memref<1xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb4
    %6 = cmpi "slt", %5, %c20_i32 : i32
    %7 = index_cast %5 : i32 to index
    cond_br %6, ^bb2(%c0_i32 : i32), ^bb5(%c0_i32 : i32)
  ^bb2(%8: i32):  // 2 preds: ^bb1, ^bb3
    %9 = cmpi "slt", %8, %c25_i32 : i32
    %10 = index_cast %8 : i32 to index
    cond_br %9, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %11 = muli %5, %8 : i32
    %12 = addi %11, %c1_i32 : i32
    %13 = remi_signed %12, %c20_i32 : i32
    %14 = sitofp %13 : i32 to f64
    %15 = sitofp %c20_i32 : i32 to f64
    %16 = divf %14, %15 : f64
    store %16, %2[%7, %10] : memref<20x25xf64>
    %17 = addi %8, %c1_i32 : i32
    br ^bb2(%17 : i32)
  ^bb4:  // pred: ^bb2
    %18 = addi %5, %c1_i32 : i32
    br ^bb1(%18 : i32)
  ^bb5(%19: i32):  // 2 preds: ^bb1, ^bb8
    %20 = cmpi "slt", %19, %c20_i32 : i32
    %21 = index_cast %19 : i32 to index
    cond_br %20, ^bb6(%c0_i32 : i32), ^bb9(%c0_i32 : i32)
  ^bb6(%22: i32):  // 2 preds: ^bb5, ^bb7
    %23 = cmpi "slt", %22, %c30_i32 : i32
    %24 = index_cast %22 : i32 to index
    cond_br %23, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %25 = addi %22, %c1_i32 : i32
    %26 = muli %19, %25 : i32
    %27 = remi_signed %26, %c30_i32 : i32
    %28 = sitofp %27 : i32 to f64
    %29 = sitofp %c30_i32 : i32 to f64
    %30 = divf %28, %29 : f64
    store %30, %3[%21, %24] : memref<20x30xf64>
    br ^bb6(%25 : i32)
  ^bb8:  // pred: ^bb6
    %31 = addi %19, %c1_i32 : i32
    br ^bb5(%31 : i32)
  ^bb9(%32: i32):  // 2 preds: ^bb5, ^bb13
    %33 = cmpi "slt", %32, %c30_i32 : i32
    %34 = index_cast %32 : i32 to index
    cond_br %33, ^bb11(%c0_i32 : i32), ^bb10
  ^bb10:  // pred: ^bb9
    %35 = load %0[%c0] : memref<1xf64>
    %36 = load %1[%c0] : memref<1xf64>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
          affine.for %arg5 = #map0(%arg3) to min #map2(%arg3) {
            call @S0(%2, %arg4, %arg5, %36) : (memref<20x25xf64>, index, index, f64) -> ()
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 1 {
          affine.for %arg5 = #map0(%arg2) to min #map3(%arg2) {
            affine.for %arg6 = #map0(%arg4) to min #map1(%arg4) {
              affine.for %arg7 = #map0(%arg3) to min #map2(%arg3) {
                call @S1(%2, %arg6, %arg7, %4, %arg5, %35, %3) : (memref<20x25xf64>, index, index, memref<30x25xf64>, index, f64, memref<20x30xf64>) -> ()
              }
            }
          }
        }
      }
    }
    call @print_array(%c20_i32, %c25_i32, %2) : (i32, i32, memref<20x25xf64>) -> ()
    return %c0_i32 : i32
  ^bb11(%37: i32):  // 2 preds: ^bb9, ^bb12
    %38 = cmpi "slt", %37, %c25_i32 : i32
    %39 = index_cast %37 : i32 to index
    cond_br %38, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %40 = addi %37, %c2_i32 : i32
    %41 = muli %32, %40 : i32
    %42 = remi_signed %41, %c25_i32 : i32
    %43 = sitofp %42 : i32 to f64
    %44 = sitofp %c25_i32 : i32 to f64
    %45 = divf %43, %44 : f64
    store %45, %4[%34, %39] : memref<30x25xf64>
    %46 = addi %37, %c1_i32 : i32
    br ^bb11(%46 : i32)
  ^bb13:  // pred: ^bb11
    %47 = addi %32, %c1_i32 : i32
    br ^bb9(%47 : i32)
  }
  func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<20x25xf64>, %arg6: memref<20x30xf64>, %arg7: memref<30x25xf64>) {
    %c0 = constant 0 : index
    %cst = constant 1.500000e+00 : f64
    %cst_0 = constant 1.200000e+00 : f64
    %c0_i32 = constant 0 : i32
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    store %cst, %arg3[%c0] : memref<?xf64>
    store %cst_0, %arg4[%c0] : memref<?xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb4
    %1 = cmpi "slt", %0, %arg0 : i32
    %2 = index_cast %0 : i32 to index
    cond_br %1, ^bb2(%c0_i32 : i32), ^bb5(%c0_i32 : i32)
  ^bb2(%3: i32):  // 2 preds: ^bb1, ^bb3
    %4 = cmpi "slt", %3, %arg1 : i32
    %5 = index_cast %3 : i32 to index
    cond_br %4, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %6 = muli %0, %3 : i32
    %7 = addi %6, %c1_i32 : i32
    %8 = remi_signed %7, %arg0 : i32
    %9 = sitofp %8 : i32 to f64
    %10 = sitofp %arg0 : i32 to f64
    %11 = divf %9, %10 : f64
    store %11, %arg5[%2, %5] : memref<20x25xf64>
    %12 = addi %3, %c1_i32 : i32
    br ^bb2(%12 : i32)
  ^bb4:  // pred: ^bb2
    %13 = addi %0, %c1_i32 : i32
    br ^bb1(%13 : i32)
  ^bb5(%14: i32):  // 2 preds: ^bb1, ^bb8
    %15 = cmpi "slt", %14, %arg0 : i32
    %16 = index_cast %14 : i32 to index
    cond_br %15, ^bb6(%c0_i32 : i32), ^bb9(%c0_i32 : i32)
  ^bb6(%17: i32):  // 2 preds: ^bb5, ^bb7
    %18 = cmpi "slt", %17, %arg2 : i32
    %19 = index_cast %17 : i32 to index
    cond_br %18, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %20 = addi %17, %c1_i32 : i32
    %21 = muli %14, %20 : i32
    %22 = remi_signed %21, %arg2 : i32
    %23 = sitofp %22 : i32 to f64
    %24 = sitofp %arg2 : i32 to f64
    %25 = divf %23, %24 : f64
    store %25, %arg6[%16, %19] : memref<20x30xf64>
    br ^bb6(%20 : i32)
  ^bb8:  // pred: ^bb6
    %26 = addi %14, %c1_i32 : i32
    br ^bb5(%26 : i32)
  ^bb9(%27: i32):  // 2 preds: ^bb5, ^bb13
    %28 = cmpi "slt", %27, %arg2 : i32
    %29 = index_cast %27 : i32 to index
    cond_br %28, ^bb11(%c0_i32 : i32), ^bb10
  ^bb10:  // pred: ^bb9
    return
  ^bb11(%30: i32):  // 2 preds: ^bb9, ^bb12
    %31 = cmpi "slt", %30, %arg1 : i32
    %32 = index_cast %30 : i32 to index
    cond_br %31, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %33 = addi %30, %c2_i32 : i32
    %34 = muli %27, %33 : i32
    %35 = remi_signed %34, %arg1 : i32
    %36 = sitofp %35 : i32 to f64
    %37 = sitofp %arg1 : i32 to f64
    %38 = divf %36, %37 : f64
    store %38, %arg7[%29, %32] : memref<30x25xf64>
    %39 = addi %30, %c1_i32 : i32
    br ^bb11(%39 : i32)
  ^bb13:  // pred: ^bb11
    %40 = addi %27, %c1_i32 : i32
    br ^bb9(%40 : i32)
  }
  func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<20x25xf64>, %arg6: memref<20x30xf64>, %arg7: memref<30x25xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        call @S0(%arg5, %arg8, %arg9, %arg4) : (memref<20x25xf64>, index, index, f64) -> ()
      }
      affine.for %arg9 = 0 to %2 {
        affine.for %arg10 = 0 to %1 {
          call @S1(%arg5, %arg8, %arg10, %arg7, %arg9, %arg3, %arg6) : (memref<20x25xf64>, index, index, memref<30x25xf64>, index, f64, memref<20x30xf64>) -> ()
        }
      }
    }
    return
  }
  func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<20x25xf64>) {
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
    %29 = cmpi "slt", %28, %arg1 : i32
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
    %39 = load %arg2[%15, %30] : memref<20x25xf64>
    %40 = llvm.mlir.cast %39 : f64 to !llvm.double
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %42 = addi %28, %c1_i32 : i32
    br ^bb3(%42 : i32)
  ^bb5:  // pred: ^bb3
    %43 = addi %13, %c1_i32 : i32
    br ^bb1(%43 : i32)
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<20x25xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<20x25xf64>
    %1 = mulf %0, %arg3 : f64
    affine.store %1, %arg0[%arg1, %arg2] : memref<20x25xf64>
    return
  }
  func private @S1(%arg0: memref<20x25xf64>, %arg1: index, %arg2: index, %arg3: memref<30x25xf64>, %arg4: index, %arg5: f64, %arg6: memref<20x30xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<20x25xf64>
    %1 = affine.load %arg6[%arg1, %arg4] : memref<20x30xf64>
    %2 = mulf %arg5, %1 : f64
    %3 = affine.load %arg3[%arg4, %arg2] : memref<30x25xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[%arg1, %arg2] : memref<20x25xf64>
    return
  }
  func @kernel_gemm_new(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<20x25xf64>, %arg6: memref<20x30xf64>, %arg7: memref<30x25xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg0 : i32 to index
    affine.for %arg8 = 0 to #map4()[%2] {
      affine.for %arg9 = 0 to #map4()[%1] {
        affine.for %arg10 = #map0(%arg8) to min #map5(%arg8)[%2] {
          affine.for %arg11 = #map0(%arg9) to min #map5(%arg9)[%1] {
            call @S0(%arg5, %arg10, %arg11, %arg4) : (memref<20x25xf64>, index, index, f64) -> ()
          }
        }
      }
    }
    affine.for %arg8 = 0 to #map4()[%0] {
      affine.for %arg9 = 0 to #map4()[%1] {
        affine.for %arg10 = 0 to #map4()[%2] {
          affine.for %arg11 = #map0(%arg8) to min #map5(%arg8)[%0] {
            affine.for %arg12 = #map0(%arg10) to min #map5(%arg10)[%2] {
              affine.for %arg13 = #map0(%arg9) to min #map5(%arg9)[%1] {
                call @S1(%arg5, %arg12, %arg13, %arg7, %arg11, %arg3, %arg6) : (memref<20x25xf64>, index, index, memref<30x25xf64>, index, f64, memref<20x30xf64>) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}

