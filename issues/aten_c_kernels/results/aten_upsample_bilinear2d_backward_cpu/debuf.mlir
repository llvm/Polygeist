#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0 + d1 * 56 + d2 * 8)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 5.000000e+00 : f32
    %cst_5 = arith.constant 8.000000e+00 : f32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.addf %9, %cst_0 : f32
        %11 = arith.mulf %10, %cst_1 : f32
        %12 = arith.divf %11, %cst_2 : f32
        %13 = arith.subf %12, %cst_0 : f32
        %14 = arith.cmpf olt, %13, %cst : f32
        %15 = arith.select %14, %cst, %13 : f32
        %16 = arith.fptosi %15 : f32 to i32
        %17 = arith.addi %6, %16 : i32
        %18 = arith.muli %17, %c5_i32 : i32
        %19 = arith.sitofp %16 : i32 to f32
        %20 = arith.subf %15, %19 : f32
        %21 = arith.subf %cst_3, %20 : f32
        %22 = arith.addi %16, %c1_i32 : i32
        %23 = arith.cmpi slt, %22, %c4_i32 : i32
        %24 = arith.select %23, %22, %16 : i32
        %25 = arith.addi %6, %24 : i32
        %26 = arith.muli %25, %c5_i32 : i32
        %27 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %28 = arith.index_cast %arg6 : index to i32
          %29 = arith.sitofp %28 : i32 to f32
          %30 = arith.addf %29, %cst_0 : f32
          %31 = arith.mulf %30, %cst_4 : f32
          %32 = arith.divf %31, %cst_5 : f32
          %33 = arith.subf %32, %cst_0 : f32
          %34 = arith.cmpf olt, %33, %cst : f32
          %35 = arith.select %34, %cst, %33 : f32
          %36 = arith.fptosi %35 : f32 to i32
          %37 = arith.addi %18, %36 : i32
          %38 = arith.index_cast %37 : i32 to index
          %39 = affine.apply #map1(%arg6, %arg2, %arg4)
          %extracted = tensor.extract %1[%39] : tensor<?xf32>
          %40 = arith.mulf %extracted, %21 : f32
          %41 = arith.sitofp %36 : i32 to f32
          %42 = arith.subf %35, %41 : f32
          %43 = arith.subf %cst_3, %42 : f32
          %44 = arith.mulf %40, %43 : f32
          %extracted_6 = tensor.extract %arg7[%38] : tensor<?xf32>
          %45 = arith.addf %extracted_6, %44 : f32
          %inserted = tensor.insert %45 into %arg7[%38] : tensor<?xf32>
          %46 = arith.addi %36, %c1_i32 : i32
          %47 = arith.cmpi slt, %46, %c5_i32 : i32
          %48 = arith.select %47, %46, %36 : i32
          %49 = arith.addi %18, %48 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = affine.apply #map1(%arg6, %arg2, %arg4)
          %extracted_7 = tensor.extract %1[%51] : tensor<?xf32>
          %52 = arith.mulf %extracted_7, %21 : f32
          %53 = arith.mulf %52, %42 : f32
          %extracted_8 = tensor.extract %inserted[%50] : tensor<?xf32>
          %54 = arith.addf %extracted_8, %53 : f32
          %inserted_9 = tensor.insert %54 into %inserted[%50] : tensor<?xf32>
          %55 = arith.addi %26, %36 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = affine.apply #map1(%arg6, %arg2, %arg4)
          %extracted_10 = tensor.extract %1[%57] : tensor<?xf32>
          %58 = arith.mulf %extracted_10, %20 : f32
          %59 = arith.mulf %58, %43 : f32
          %extracted_11 = tensor.extract %inserted_9[%56] : tensor<?xf32>
          %60 = arith.addf %extracted_11, %59 : f32
          %inserted_12 = tensor.insert %60 into %inserted_9[%56] : tensor<?xf32>
          %61 = arith.addi %26, %48 : i32
          %62 = arith.index_cast %61 : i32 to index
          %63 = affine.apply #map1(%arg6, %arg2, %arg4)
          %extracted_13 = tensor.extract %1[%63] : tensor<?xf32>
          %64 = arith.mulf %extracted_13, %20 : f32
          %65 = arith.mulf %64, %42 : f32
          %extracted_14 = tensor.extract %inserted_12[%62] : tensor<?xf32>
          %66 = arith.addf %extracted_14, %65 : f32
          %inserted_15 = tensor.insert %66 into %inserted_12[%62] : tensor<?xf32>
          affine.yield %inserted_15 : tensor<?xf32>
        }
        affine.yield %27 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

