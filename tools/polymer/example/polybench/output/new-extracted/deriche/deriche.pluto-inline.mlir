#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (4096, d0 * 32 + 32)>
#map2 = affine_map<(d0) -> (2160, d0 * 32 + 32)>
#map3 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str4("%0.2f \00")
  llvm.mlir.global internal constant @str3("\0A\00")
  llvm.mlir.global internal constant @str2("imgOut\00")
  llvm.mlir.global internal constant @str1("begin dump: %s\00")
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c4096_i32 = constant 4096 : i32
    %c2160_i32 = constant 2160 : i32
    %c0 = constant 0 : index
    %cst = constant 2.500000e-01 : f64
    %c0_i32 = constant 0 : i32
    %c313_i32 = constant 313 : i32
    %c991_i32 = constant 991 : i32
    %c65536_i32 = constant 65536 : i32
    %cst_0 = constant 6.553500e+04 : f32
    %cst_1 = constant 1.000000e+00 : f32
    %cst_2 = constant 2.000000e+00 : f32
    %c1_i32 = constant 1 : i32
    %c2160 = constant 2160 : index
    %c4096 = constant 4096 : index
    %0 = alloca() : memref<1xf32>
    %1 = alloc() : memref<4096x2160xf32>
    %2 = alloc() : memref<4096x2160xf32>
    %3 = alloc() : memref<4096x2160xf32>
    %4 = alloc() : memref<4096x2160xf32>
    %5 = fptrunc %cst : f64 to f32
    store %5, %0[%c0] : memref<1xf32>
    br ^bb1(%c0_i32 : i32)
  ^bb1(%6: i32):  // 2 preds: ^bb0, ^bb5
    %7 = cmpi "slt", %6, %c4096_i32 : i32
    %8 = index_cast %6 : i32 to index
    cond_br %7, ^bb3(%c0_i32 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %9 = load %0[%c0] : memref<1xf32>
    %10 = alloca() : memref<1xf32>
    %11 = alloca() : memref<1xf32>
    %12 = alloca() : memref<1xf32>
    %13 = alloca() : memref<1xf32>
    %14 = alloca() : memref<1xf32>
    %15 = alloca() : memref<1xf32>
    %16 = alloca() : memref<1xf32>
    %17 = alloca() : memref<1xf32>
    %18 = alloca() : memref<1xf32>
    %19 = alloca() : memref<1xf32>
    %20 = alloca() : memref<1xf32>
    %21 = alloca() : memref<1xf32>
    %22 = alloca() : memref<1xf32>
    %23 = alloca() : memref<1xf32>
    %24 = alloca() : memref<1xf32>
    %25 = alloca() : memref<1xf32>
    %26 = alloca() : memref<1xf32>
    affine.for %arg2 = 0 to 4096 {
      call @S17(%19) : (memref<1xf32>) -> ()
      call @S16(%26) : (memref<1xf32>) -> ()
      call @S15(%20) : (memref<1xf32>) -> ()
      call @S14(%21) : (memref<1xf32>) -> ()
      affine.for %arg3 = 0 to 2160 {
        call @S18(%4, %arg2, %arg3, %c2160, %20, %13, %21, %15, %19, %24, %26, %25) : (memref<4096x2160xf32>, index, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
        call @S21(%20, %21) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S22(%21, %4, %arg2, %arg3, %c2160) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
        call @S19(%19, %26) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S20(%26, %1, %arg2, %arg3, %c2160) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
      }
    }
    %27 = negf %9 : f32
    %28 = exp %27 : f32
    %29 = subf %cst_1, %28 : f32
    %30 = mulf %29, %29 : f32
    %31 = mulf %cst_2, %9 : f32
    %32 = mulf %31, %28 : f32
    %33 = addf %cst_1, %32 : f32
    %34 = exp %31 : f32
    %35 = subf %33, %34 : f32
    %36 = divf %30, %35 : f32
    %37 = negf %36 : f32
    %38 = negf %cst_2 : f32
    %39 = mulf %38, %9 : f32
    %40 = exp %39 : f32
    %41 = mulf %37, %40 : f32
    affine.store %41, %24[0] : memref<1xf32>
    %42 = mulf %cst_2, %9 : f32
    %43 = exp %42 : f32
    %44 = negf %9 : f32
    %45 = exp %44 : f32
    %46 = mulf %42, %45 : f32
    %47 = subf %cst_1, %45 : f32
    %48 = mulf %47, %47 : f32
    %49 = addf %cst_1, %46 : f32
    %50 = subf %49, %43 : f32
    %51 = divf %48, %50 : f32
    %52 = mulf %51, %45 : f32
    %53 = addf %9, %cst_1 : f32
    %54 = mulf %52, %53 : f32
    affine.store %54, %25[0] : memref<1xf32>
    %55 = mulf %cst_2, %9 : f32
    %56 = exp %55 : f32
    %57 = negf %9 : f32
    %58 = exp %57 : f32
    %59 = mulf %55, %58 : f32
    %60 = subf %cst_1, %58 : f32
    %61 = mulf %60, %60 : f32
    %62 = addf %cst_1, %59 : f32
    %63 = subf %62, %56 : f32
    %64 = divf %61, %63 : f32
    %65 = mulf %64, %58 : f32
    %66 = subf %9, %cst_1 : f32
    %67 = mulf %65, %66 : f32
    affine.store %67, %17[0] : memref<1xf32>
    %68 = negf %9 : f32
    %69 = exp %68 : f32
    %70 = subf %cst_1, %69 : f32
    %71 = mulf %70, %70 : f32
    %72 = mulf %cst_2, %9 : f32
    %73 = mulf %72, %69 : f32
    %74 = addf %cst_1, %73 : f32
    %75 = exp %72 : f32
    %76 = subf %74, %75 : f32
    %77 = divf %71, %76 : f32
    affine.store %77, %18[0] : memref<1xf32>
    affine.for %arg2 = 0 to 4096 {
      affine.for %arg3 = 0 to 2160 {
        call @S10(%3, %arg2, %arg3, %12, %13, %14, %15, %16, %17, %1, %18) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S12(%12, %14) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S13(%14, %3, %arg2, %arg3) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S11(%16, %1, %arg2, %arg3) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
      call @S9(%16) : (memref<1xf32>) -> ()
      call @S8(%12) : (memref<1xf32>) -> ()
      call @S7(%14) : (memref<1xf32>) -> ()
    }
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 68 {
        affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
          affine.for %arg5 = #map0(%arg3) to min #map2(%arg3) {
            call @S23(%2, %arg4, %arg5, %4, %3, %23) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
          }
        }
      }
    }
    affine.for %arg2 = 0 to 4096 {
      call @S34(%20) : (memref<1xf32>) -> ()
      call @S33(%21) : (memref<1xf32>) -> ()
      call @S32(%10) : (memref<1xf32>) -> ()
      call @S31(%11) : (memref<1xf32>) -> ()
      affine.for %arg3 = 0 to 2160 {
        call @S35(%4, %arg2, %arg3, %c4096, %20, %13, %21, %15, %10, %24, %11, %25) : (memref<4096x2160xf32>, index, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
        call @S38(%20, %21) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S39(%21, %4, %arg2, %arg3, %c4096) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
        call @S36(%10, %11) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S37(%11, %2, %arg2, %arg3, %c4096) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
      }
    }
    affine.for %arg2 = 0 to 4096 {
      call @S26(%12) : (memref<1xf32>) -> ()
      call @S25(%14) : (memref<1xf32>) -> ()
      call @S24(%22) : (memref<1xf32>) -> ()
      affine.for %arg3 = 0 to 2160 {
        call @S27(%3, %arg2, %arg3, %12, %13, %14, %15, %22, %17, %2, %18) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S29(%12, %14) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S30(%14, %3, %arg2, %arg3) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S28(%22, %2, %arg2, %arg3) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    %78 = negf %cst_2 : f32
    %79 = mulf %78, %9 : f32
    %80 = exp %79 : f32
    %81 = negf %80 : f32
    affine.store %81, %13[0] : memref<1xf32>
    call @S4(%15, %9) : (memref<1xf32>, f32) -> ()
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 68 {
        affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
          affine.for %arg5 = #map0(%arg3) to min #map2(%arg3) {
            call @S40(%2, %arg4, %arg5, %4, %3, %23) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
          }
        }
      }
    }
    %82 = sitofp %c1_i32 : i32 to f32
    affine.store %82, %23[0] : memref<1xf32>
    call @print_array(%c4096_i32, %c2160_i32, %2) : (i32, i32, memref<4096x2160xf32>) -> ()
    return %c0_i32 : i32
  ^bb3(%83: i32):  // 2 preds: ^bb1, ^bb4
    %84 = cmpi "slt", %83, %c2160_i32 : i32
    %85 = index_cast %83 : i32 to index
    cond_br %84, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %86 = muli %6, %c313_i32 : i32
    %87 = muli %83, %c991_i32 : i32
    %88 = addi %86, %87 : i32
    %89 = remi_signed %88, %c65536_i32 : i32
    %90 = sitofp %89 : i32 to f32
    %91 = divf %90, %cst_0 : f32
    store %91, %1[%8, %85] : memref<4096x2160xf32>
    %92 = addi %83, %c1_i32 : i32
    br ^bb3(%92 : i32)
  ^bb5:  // pred: ^bb3
    %93 = addi %6, %c1_i32 : i32
    br ^bb1(%93 : i32)
  }
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<4096x2160xf32>) {
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
    %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<7 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
    %20 = llvm.mlir.addressof @str2 : !llvm.ptr<array<7 x i8>>
    %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
    %31 = muli %13, %arg1 : i32
    %32 = addi %31, %28 : i32
    %33 = remi_signed %32, %c20_i32 : i32
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
    %37 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg2[%15, %30] : memref<4096x2160xf32>
    %40 = fpext %39 : f32 to f64
    %41 = llvm.mlir.cast %40 : f64 to !llvm.double
    %42 = llvm.call @fprintf(%36, %38, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    %43 = addi %28, %c1_i32 : i32
    br ^bb3(%43 : i32)
  ^bb5:  // pred: ^bb3
    %44 = addi %13, %c1_i32 : i32
    br ^bb1(%44 : i32)
  }
  func private @free(memref<?xi8>)
  func private @S0(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    %0 = negf %arg1 : f32
    %1 = exp %0 : f32
    %2 = subf %cst_0, %1 : f32
    %3 = mulf %2, %2 : f32
    %4 = mulf %cst, %arg1 : f32
    %5 = mulf %4, %1 : f32
    %6 = addf %cst_0, %5 : f32
    %7 = exp %4 : f32
    %8 = subf %6, %7 : f32
    %9 = divf %3, %8 : f32
    affine.store %9, %arg0[0] : memref<1xf32>
    return
  }
  func private @S1(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    %0 = mulf %cst, %arg1 : f32
    %1 = exp %0 : f32
    %2 = negf %arg1 : f32
    %3 = exp %2 : f32
    %4 = mulf %0, %3 : f32
    %5 = subf %cst_0, %3 : f32
    %6 = mulf %5, %5 : f32
    %7 = addf %cst_0, %4 : f32
    %8 = subf %7, %1 : f32
    %9 = divf %6, %8 : f32
    %10 = mulf %9, %3 : f32
    %11 = subf %arg1, %cst_0 : f32
    %12 = mulf %10, %11 : f32
    affine.store %12, %arg0[0] : memref<1xf32>
    return
  }
  func private @S2(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    %0 = mulf %cst, %arg1 : f32
    %1 = exp %0 : f32
    %2 = negf %arg1 : f32
    %3 = exp %2 : f32
    %4 = mulf %0, %3 : f32
    %5 = subf %cst_0, %3 : f32
    %6 = mulf %5, %5 : f32
    %7 = addf %cst_0, %4 : f32
    %8 = subf %7, %1 : f32
    %9 = divf %6, %8 : f32
    %10 = mulf %9, %3 : f32
    %11 = addf %arg1, %cst_0 : f32
    %12 = mulf %10, %11 : f32
    affine.store %12, %arg0[0] : memref<1xf32>
    return
  }
  func private @S3(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %0 = negf %arg1 : f32
    %1 = exp %0 : f32
    %2 = subf %cst, %1 : f32
    %3 = mulf %2, %2 : f32
    %4 = mulf %cst_0, %arg1 : f32
    %5 = mulf %4, %1 : f32
    %6 = addf %cst, %5 : f32
    %7 = exp %4 : f32
    %8 = subf %6, %7 : f32
    %9 = divf %3, %8 : f32
    %10 = negf %9 : f32
    %11 = negf %cst_0 : f32
    %12 = mulf %11, %arg1 : f32
    %13 = exp %12 : f32
    %14 = mulf %10, %13 : f32
    affine.store %14, %arg0[0] : memref<1xf32>
    return
  }
  func private @S4(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = llvm.mlir.cast %cst : f32 to !llvm.float
    %1 = negf %arg1 : f32
    %2 = llvm.mlir.cast %1 : f32 to !llvm.float
    %3 = "llvm.intr.pow"(%0, %2) : (!llvm.float, !llvm.float) -> !llvm.float
    %4 = llvm.mlir.cast %3 : !llvm.float to f32
    affine.store %4, %arg0[0] : memref<1xf32>
    return
  }
  func private @S5(%arg0: memref<1xf32>, %arg1: f32) attributes {scop.stmt} {
    %cst = constant 2.000000e+00 : f32
    %0 = negf %cst : f32
    %1 = mulf %0, %arg1 : f32
    %2 = exp %1 : f32
    %3 = negf %2 : f32
    affine.store %3, %arg0[0] : memref<1xf32>
    return
  }
  func private @S6(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = sitofp %c1_i32 : i32 to f32
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S7(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S8(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S9(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S10(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<1xf32>, %arg4: memref<1xf32>, %arg5: memref<1xf32>, %arg6: memref<1xf32>, %arg7: memref<1xf32>, %arg8: memref<1xf32>, %arg9: memref<4096x2160xf32>, %arg10: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg10[0] : memref<1xf32>
    %1 = affine.load %arg9[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %arg8[0] : memref<1xf32>
    %4 = affine.load %arg7[0] : memref<1xf32>
    %5 = mulf %3, %4 : f32
    %6 = addf %2, %5 : f32
    %7 = affine.load %arg6[0] : memref<1xf32>
    %8 = affine.load %arg5[0] : memref<1xf32>
    %9 = mulf %7, %8 : f32
    %10 = addf %6, %9 : f32
    %11 = affine.load %arg4[0] : memref<1xf32>
    %12 = affine.load %arg3[0] : memref<1xf32>
    %13 = mulf %11, %12 : f32
    %14 = addf %10, %13 : f32
    affine.store %14, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func private @S11(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S12(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S13(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S14(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S15(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S16(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S17(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S18(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<1xf32>, %arg5: memref<1xf32>, %arg6: memref<1xf32>, %arg7: memref<1xf32>, %arg8: memref<1xf32>, %arg9: memref<1xf32>, %arg10: memref<1xf32>, %arg11: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg11[0] : memref<1xf32>
    %1 = affine.load %arg10[0] : memref<1xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %arg9[0] : memref<1xf32>
    %4 = affine.load %arg8[0] : memref<1xf32>
    %5 = mulf %3, %4 : f32
    %6 = addf %2, %5 : f32
    %7 = affine.load %arg7[0] : memref<1xf32>
    %8 = affine.load %arg6[0] : memref<1xf32>
    %9 = mulf %7, %8 : f32
    %10 = addf %6, %9 : f32
    %11 = affine.load %arg5[0] : memref<1xf32>
    %12 = affine.load %arg4[0] : memref<1xf32>
    %13 = mulf %11, %12 : f32
    %14 = addf %10, %13 : f32
    affine.store %14, %arg0[%arg1, -%arg2 + symbol(%arg3) - 1] : memref<4096x2160xf32>
    return
  }
  func private @S19(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S20(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, -%arg3 + symbol(%arg4) - 1] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S21(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S22(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, -%arg3 + symbol(%arg4) - 1] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S23(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf32>
    %1 = affine.load %arg4[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = affine.load %arg3[%arg1, %arg2] : memref<4096x2160xf32>
    %3 = addf %1, %2 : f32
    %4 = mulf %0, %3 : f32
    affine.store %4, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func private @S24(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S25(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S26(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S27(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<1xf32>, %arg4: memref<1xf32>, %arg5: memref<1xf32>, %arg6: memref<1xf32>, %arg7: memref<1xf32>, %arg8: memref<1xf32>, %arg9: memref<4096x2160xf32>, %arg10: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg10[0] : memref<1xf32>
    %1 = affine.load %arg9[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %arg8[0] : memref<1xf32>
    %4 = affine.load %arg7[0] : memref<1xf32>
    %5 = mulf %3, %4 : f32
    %6 = addf %2, %5 : f32
    %7 = affine.load %arg6[0] : memref<1xf32>
    %8 = affine.load %arg5[0] : memref<1xf32>
    %9 = mulf %7, %8 : f32
    %10 = addf %6, %9 : f32
    %11 = affine.load %arg4[0] : memref<1xf32>
    %12 = affine.load %arg3[0] : memref<1xf32>
    %13 = mulf %11, %12 : f32
    %14 = addf %10, %13 : f32
    affine.store %14, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func private @S28(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S29(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S30(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S31(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S32(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S33(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S34(%arg0: memref<1xf32>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg0[0] : memref<1xf32>
    return
  }
  func private @S35(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<1xf32>, %arg5: memref<1xf32>, %arg6: memref<1xf32>, %arg7: memref<1xf32>, %arg8: memref<1xf32>, %arg9: memref<1xf32>, %arg10: memref<1xf32>, %arg11: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg11[0] : memref<1xf32>
    %1 = affine.load %arg10[0] : memref<1xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %arg9[0] : memref<1xf32>
    %4 = affine.load %arg8[0] : memref<1xf32>
    %5 = mulf %3, %4 : f32
    %6 = addf %2, %5 : f32
    %7 = affine.load %arg7[0] : memref<1xf32>
    %8 = affine.load %arg6[0] : memref<1xf32>
    %9 = mulf %7, %8 : f32
    %10 = addf %6, %9 : f32
    %11 = affine.load %arg5[0] : memref<1xf32>
    %12 = affine.load %arg4[0] : memref<1xf32>
    %13 = mulf %11, %12 : f32
    %14 = addf %10, %13 : f32
    affine.store %14, %arg0[-%arg1 + symbol(%arg3) - 1, %arg2] : memref<4096x2160xf32>
    return
  }
  func private @S36(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S37(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[-%arg2 + symbol(%arg4) - 1, %arg3] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S38(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S39(%arg0: memref<1xf32>, %arg1: memref<4096x2160xf32>, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[-%arg2 + symbol(%arg4) - 1, %arg3] : memref<4096x2160xf32>
    affine.store %0, %arg0[0] : memref<1xf32>
    return
  }
  func private @S40(%arg0: memref<4096x2160xf32>, %arg1: index, %arg2: index, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<1xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf32>
    %1 = affine.load %arg4[%arg1, %arg2] : memref<4096x2160xf32>
    %2 = affine.load %arg3[%arg1, %arg2] : memref<4096x2160xf32>
    %3 = addf %1, %2 : f32
    %4 = mulf %0, %3 : f32
    affine.store %4, %arg0[%arg1, %arg2] : memref<4096x2160xf32>
    return
  }
  func @"\00\00\00\00\00\00\00\00\1002\03\00\00\00\00ew"(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %c1_i32 = constant 1 : i32
    %0 = alloca() : memref<1xf32>
    %1 = alloca() : memref<1xf32>
    %2 = alloca() : memref<1xf32>
    %3 = alloca() : memref<1xf32>
    %4 = alloca() : memref<1xf32>
    %5 = alloca() : memref<1xf32>
    %6 = alloca() : memref<1xf32>
    %7 = alloca() : memref<1xf32>
    %8 = alloca() : memref<1xf32>
    %9 = alloca() : memref<1xf32>
    %10 = alloca() : memref<1xf32>
    %11 = alloca() : memref<1xf32>
    %12 = alloca() : memref<1xf32>
    %13 = alloca() : memref<1xf32>
    %14 = alloca() : memref<1xf32>
    %15 = alloca() : memref<1xf32>
    %16 = alloca() : memref<1xf32>
    %17 = index_cast %arg1 : i32 to index
    %18 = index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %18 {
      call @S17(%9) : (memref<1xf32>) -> ()
      call @S16(%16) : (memref<1xf32>) -> ()
      call @S15(%10) : (memref<1xf32>) -> ()
      call @S14(%11) : (memref<1xf32>) -> ()
      affine.for %arg8 = 0 to %17 {
        call @S18(%arg6, %arg7, %arg8, %17, %10, %3, %11, %5, %9, %14, %16, %15) : (memref<4096x2160xf32>, index, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
        call @S21(%10, %11) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S22(%11, %arg6, %arg7, %arg8, %17) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
        call @S19(%9, %16) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S20(%16, %arg3, %arg7, %arg8, %17) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
      }
    }
    %19 = negf %arg2 : f32
    %20 = exp %19 : f32
    %21 = subf %cst, %20 : f32
    %22 = mulf %21, %21 : f32
    %23 = mulf %cst_0, %arg2 : f32
    %24 = mulf %23, %20 : f32
    %25 = addf %cst, %24 : f32
    %26 = exp %23 : f32
    %27 = subf %25, %26 : f32
    %28 = divf %22, %27 : f32
    %29 = negf %28 : f32
    %30 = negf %cst_0 : f32
    %31 = mulf %30, %arg2 : f32
    %32 = exp %31 : f32
    %33 = mulf %29, %32 : f32
    affine.store %33, %14[0] : memref<1xf32>
    %34 = mulf %cst_0, %arg2 : f32
    %35 = exp %34 : f32
    %36 = negf %arg2 : f32
    %37 = exp %36 : f32
    %38 = mulf %34, %37 : f32
    %39 = subf %cst, %37 : f32
    %40 = mulf %39, %39 : f32
    %41 = addf %cst, %38 : f32
    %42 = subf %41, %35 : f32
    %43 = divf %40, %42 : f32
    %44 = mulf %43, %37 : f32
    %45 = addf %arg2, %cst : f32
    %46 = mulf %44, %45 : f32
    affine.store %46, %15[0] : memref<1xf32>
    %47 = mulf %cst_0, %arg2 : f32
    %48 = exp %47 : f32
    %49 = negf %arg2 : f32
    %50 = exp %49 : f32
    %51 = mulf %47, %50 : f32
    %52 = subf %cst, %50 : f32
    %53 = mulf %52, %52 : f32
    %54 = addf %cst, %51 : f32
    %55 = subf %54, %48 : f32
    %56 = divf %53, %55 : f32
    %57 = mulf %56, %50 : f32
    %58 = subf %arg2, %cst : f32
    %59 = mulf %57, %58 : f32
    affine.store %59, %7[0] : memref<1xf32>
    %60 = negf %arg2 : f32
    %61 = exp %60 : f32
    %62 = subf %cst, %61 : f32
    %63 = mulf %62, %62 : f32
    %64 = mulf %cst_0, %arg2 : f32
    %65 = mulf %64, %61 : f32
    %66 = addf %cst, %65 : f32
    %67 = exp %64 : f32
    %68 = subf %66, %67 : f32
    %69 = divf %63, %68 : f32
    affine.store %69, %8[0] : memref<1xf32>
    affine.for %arg7 = 0 to %18 {
      affine.for %arg8 = 0 to %17 {
        call @S10(%arg5, %arg7, %arg8, %2, %3, %4, %5, %6, %7, %arg3, %8) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S12(%2, %4) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S13(%4, %arg5, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S11(%6, %arg3, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
      call @S9(%6) : (memref<1xf32>) -> ()
      call @S8(%2) : (memref<1xf32>) -> ()
      call @S7(%4) : (memref<1xf32>) -> ()
    }
    affine.for %arg7 = 0 to #map3()[%18] {
      affine.for %arg8 = 0 to #map3()[%17] {
        affine.for %arg9 = #map0(%arg7) to min #map4(%arg7)[%18] {
          affine.for %arg10 = #map0(%arg8) to min #map4(%arg8)[%17] {
            call @S23(%arg4, %arg9, %arg10, %arg6, %arg5, %13) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
          }
        }
      }
    }
    affine.for %arg7 = 0 to %18 {
      call @S34(%10) : (memref<1xf32>) -> ()
      call @S33(%11) : (memref<1xf32>) -> ()
      call @S32(%0) : (memref<1xf32>) -> ()
      call @S31(%1) : (memref<1xf32>) -> ()
      affine.for %arg8 = 0 to %17 {
        call @S35(%arg6, %arg7, %arg8, %18, %10, %3, %11, %5, %0, %14, %1, %15) : (memref<4096x2160xf32>, index, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
        call @S38(%10, %11) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S39(%11, %arg6, %arg7, %arg8, %18) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
        call @S36(%0, %1) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S37(%1, %arg4, %arg7, %arg8, %18) : (memref<1xf32>, memref<4096x2160xf32>, index, index, index) -> ()
      }
    }
    affine.for %arg7 = 0 to %18 {
      call @S26(%2) : (memref<1xf32>) -> ()
      call @S25(%4) : (memref<1xf32>) -> ()
      call @S24(%12) : (memref<1xf32>) -> ()
      affine.for %arg8 = 0 to %17 {
        call @S27(%arg5, %arg7, %arg8, %2, %3, %4, %5, %12, %7, %arg4, %8) : (memref<4096x2160xf32>, index, index, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
        call @S29(%2, %4) : (memref<1xf32>, memref<1xf32>) -> ()
        call @S30(%4, %arg5, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
        call @S28(%12, %arg4, %arg7, %arg8) : (memref<1xf32>, memref<4096x2160xf32>, index, index) -> ()
      }
    }
    %70 = negf %cst_0 : f32
    %71 = mulf %70, %arg2 : f32
    %72 = exp %71 : f32
    %73 = negf %72 : f32
    affine.store %73, %3[0] : memref<1xf32>
    call @S4(%5, %arg2) : (memref<1xf32>, f32) -> ()
    affine.for %arg7 = 0 to #map3()[%18] {
      affine.for %arg8 = 0 to #map3()[%17] {
        affine.for %arg9 = #map0(%arg7) to min #map4(%arg7)[%18] {
          affine.for %arg10 = #map0(%arg8) to min #map4(%arg8)[%17] {
            call @S40(%arg4, %arg9, %arg10, %arg6, %arg5, %13) : (memref<4096x2160xf32>, index, index, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<1xf32>) -> ()
          }
        }
      }
    }
    %74 = sitofp %c1_i32 : i32 to f32
    affine.store %74, %13[0] : memref<1xf32>
    return
  }
}

