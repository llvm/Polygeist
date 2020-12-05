#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map2 = affine_map<(d0) -> (d0 * 32)>
#map3 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map4 = affine_map<(d0, d1) -> (d0 * 32, d1)>
#map5 = affine_map<()[s0, s1] -> ((s0 + s1 - 2) floordiv 32 + 1)>
#map6 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map7 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map8 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32)>
#map9 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2lf \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("cov\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c0 = constant 0 : index
    %c32_i32 = constant 32 : i32
    %c28_i32 = constant 28 : i32
    %c0_i32 = constant 0 : i32
    %0 = alloca() : memref<1xf64>
    %1 = alloc() : memref<32x28xf64>
    %2 = alloc() : memref<28x28xf64>
    %3 = alloc() : memref<28xf64>
    %4 = memref_cast %0 : memref<1xf64> to memref<?xf64>
    call @init_array(%c28_i32, %c32_i32, %4, %1) : (i32, i32, memref<?xf64>, memref<32x28xf64>) -> ()
    %5 = load %0[%c0] : memref<1xf64>
    call @kernel_covariance_new(%c28_i32, %c32_i32, %5, %1, %2, %3) : (i32, i32, f64, memref<32x28xf64>, memref<28x28xf64>, memref<28xf64>) -> ()
    call @print_array(%c28_i32, %2) : (i32, memref<28x28xf64>) -> ()
    return %c0_i32 : i32
  }
  func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<32x28xf64>) {
    %c0 = constant 0 : index
    %c0_i32 = constant 0 : i32
    %c32_i32 = constant 32 : i32
    %c28_i32 = constant 28 : i32
    %c1_i32 = constant 1 : i32
    %0 = sitofp %arg1 : i32 to f64
    store %0, %arg2[%c0] : memref<?xf64>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb5
    %2 = cmpi "slt", %1, %c32_i32 : i32
    %3 = index_cast %1 : i32 to index
    cond_br %2, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    return
  ^bb3(%4: i32):  // 2 preds: ^bb1, ^bb4
    %5 = cmpi "slt", %4, %c28_i32 : i32
    %6 = index_cast %4 : i32 to index
    cond_br %5, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %7 = sitofp %1 : i32 to f64
    %8 = sitofp %4 : i32 to f64
    %9 = mulf %7, %8 : f64
    %10 = sitofp %c28_i32 : i32 to f64
    %11 = divf %9, %10 : f64
    store %11, %arg3[%3, %6] : memref<32x28xf64>
    %12 = addi %4, %c1_i32 : i32
    br ^bb3(%12 : i32)
  ^bb5:  // pred: ^bb3
    %13 = addi %1, %c1_i32 : i32
    br ^bb1(%13 : i32)
  }
  func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<32x28xf64>, %arg4: memref<28x28xf64>, %arg5: memref<28xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %1 {
      call @S0(%arg5, %arg6) : (memref<28xf64>, index) -> ()
      affine.for %arg7 = 0 to %0 {
        call @S1(%arg5, %arg6, %arg3, %arg7) : (memref<28xf64>, index, memref<32x28xf64>, index) -> ()
      }
      call @S2(%arg5, %arg6, %arg2) : (memref<28xf64>, index, f64) -> ()
    }
    affine.for %arg6 = 0 to %0 {
      affine.for %arg7 = 0 to %1 {
        call @S3(%arg3, %arg6, %arg7, %arg5) : (memref<32x28xf64>, index, index, memref<28xf64>) -> ()
      }
    }
    affine.for %arg6 = 0 to %1 {
      affine.for %arg7 = #map0(%arg6) to %1 {
        call @S4(%arg4, %arg6, %arg7) : (memref<28x28xf64>, index, index) -> ()
        affine.for %arg8 = 0 to %0 {
          call @S5(%arg4, %arg6, %arg7, %arg3, %arg8) : (memref<28x28xf64>, index, index, memref<32x28xf64>, index) -> ()
        }
        call @S6(%arg4, %arg6, %arg7, %arg2) : (memref<28x28xf64>, index, index, f64) -> ()
        call @S7(%arg4, %arg7, %arg6) : (memref<28x28xf64>, index, index) -> ()
      }
    }
    return
  }
  func @print_array(%arg0: i32, %arg1: memref<28x28xf64>) {
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
    %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<4 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
    %20 = llvm.mlir.addressof @str2 : !llvm.ptr<array<4 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
    %39 = load %arg1[%15, %30] : memref<28x28xf64>
    %40 = llvm.mlir.cast %39 : f64 to !llvm.double
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %42 = addi %28, %c1_i32 : i32
    br ^bb3(%42 : i32)
  ^bb5:  // pred: ^bb3
    %43 = addi %13, %c1_i32 : i32
    br ^bb1(%43 : i32)
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<28xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1] : memref<28xf64>
    return
  }
  func private @S1(%arg0: memref<28xf64>, %arg1: index, %arg2: memref<32x28xf64>, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<28xf64>
    %1 = affine.load %arg2[%arg3, %arg1] : memref<32x28xf64>
    %2 = addf %0, %1 : f64
    affine.store %2, %arg0[%arg1] : memref<28xf64>
    return
  }
  func private @S2(%arg0: memref<28xf64>, %arg1: index, %arg2: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1] : memref<28xf64>
    %1 = divf %0, %arg2 : f64
    affine.store %1, %arg0[%arg1] : memref<28xf64>
    return
  }
  func private @S3(%arg0: memref<32x28xf64>, %arg1: index, %arg2: index, %arg3: memref<28xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<32x28xf64>
    %1 = affine.load %arg3[%arg2] : memref<28xf64>
    %2 = subf %0, %1 : f64
    affine.store %2, %arg0[%arg1, %arg2] : memref<32x28xf64>
    return
  }
  func private @S4(%arg0: memref<28x28xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[%arg1, %arg2] : memref<28x28xf64>
    return
  }
  func private @S5(%arg0: memref<28x28xf64>, %arg1: index, %arg2: index, %arg3: memref<32x28xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<28x28xf64>
    %1 = affine.load %arg3[%arg4, %arg1] : memref<32x28xf64>
    %2 = affine.load %arg3[%arg4, %arg2] : memref<32x28xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[%arg1, %arg2] : memref<28x28xf64>
    return
  }
  func private @S6(%arg0: memref<28x28xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    %0 = affine.load %arg0[%arg1, %arg2] : memref<28x28xf64>
    %1 = subf %arg3, %cst : f64
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[%arg1, %arg2] : memref<28x28xf64>
    return
  }
  func private @S7(%arg0: memref<28x28xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg2, %arg1] : memref<28x28xf64>
    affine.store %0, %arg0[%arg1, %arg2] : memref<28x28xf64>
    return
  }
  func private @kernel_covariance_new(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<32x28xf64>, %arg4: memref<28x28xf64>, %arg5: memref<28xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg6 = 0 to #map1()[%0] {
      affine.for %arg7 = #map0(%arg6) to #map1()[%0] {
        affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%0] {
          affine.for %arg9 = max #map4(%arg7, %arg8) to min #map3(%arg7)[%0] {
            call @S4(%arg4, %arg8, %arg9) : (memref<28x28xf64>, index, index) -> ()
          }
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%0] {
      affine.for %arg7 = #map2(%arg6) to min #map3(%arg6)[%0] {
        call @S0(%arg5, %arg7) : (memref<28xf64>, index) -> ()
      }
    }
    affine.for %arg6 = 0 to #map1()[%0] {
      affine.for %arg7 = 0 to #map1()[%1] {
        affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%0] {
          affine.for %arg9 = #map2(%arg7) to min #map3(%arg7)[%1] {
            call @S1(%arg5, %arg8, %arg3, %arg9) : (memref<28xf64>, index, memref<32x28xf64>, index) -> ()
          }
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%0] {
      affine.for %arg7 = #map2(%arg6) to min #map3(%arg6)[%0] {
        call @S2(%arg5, %arg7, %arg2) : (memref<28xf64>, index, f64) -> ()
      }
    }
    affine.for %arg6 = 0 to #map1()[%1] {
      affine.for %arg7 = 0 to #map1()[%0] {
        affine.for %arg8 = #map2(%arg7) to min #map3(%arg7)[%0] {
          affine.for %arg9 = #map2(%arg6) to min #map3(%arg6)[%1] {
            call @S3(%arg3, %arg9, %arg8, %arg5) : (memref<32x28xf64>, index, index, memref<28xf64>) -> ()
          }
        }
      }
    }
    affine.for %arg6 = 0 to #map5()[%0, %1] {
      affine.for %arg7 = max #map6(%arg6)[%1] to min #map7(%arg6)[%0] {
        affine.for %arg8 = #map0(%arg7) to #map1()[%0] {
          affine.for %arg9 = #map2(%arg7) to min #map3(%arg7)[%0] {
            affine.for %arg10 = max #map4(%arg8, %arg9) to min #map3(%arg8)[%0] {
              affine.for %arg11 = #map8(%arg6, %arg7) to min #map9(%arg6, %arg7)[%1] {
                call @S5(%arg4, %arg9, %arg10, %arg3, %arg11) : (memref<28x28xf64>, index, index, memref<32x28xf64>, index) -> ()
              }
            }
          }
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%0] {
      affine.for %arg7 = #map0(%arg6) to #map1()[%0] {
        affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%0] {
          affine.for %arg9 = max #map4(%arg7, %arg8) to min #map3(%arg7)[%0] {
            call @S6(%arg4, %arg8, %arg9, %arg2) : (memref<28x28xf64>, index, index, f64) -> ()
            call @S7(%arg4, %arg9, %arg8) : (memref<28x28xf64>, index, index) -> ()
          }
        }
      }
    }
    return
  }
}

