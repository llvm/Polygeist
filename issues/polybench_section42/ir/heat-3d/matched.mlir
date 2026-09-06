#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<?x?x?xf64>, %arg3: memref<?x?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.250000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 restrict : memref<?x?x?xf64>
    %1 = bufferization.to_tensor %arg3 restrict : memref<?x?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.apply #map()[%2]
    %4 = affine.apply #map()[%2]
    %5 = arith.subi %4, %c1 : index
    %6 = affine.apply #map()[%2]
    %7 = arith.subi %6, %c1 : index
    %8 = affine.apply #map()[%2]
    %9 = arith.subi %8, %c1 : index
    %10 = arith.subi %3, %c1 : index
    %11 = affine.apply #map()[%2]
    %12:2 = affine.for %arg4 = 1 to 501 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
      %extracted_slice = tensor.extract_slice %arg6[2, 1, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_1 = tensor.extract_slice %arg6[1, 1, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_2 = tensor.extract_slice %arg6[0, 1, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_3 = tensor.extract_slice %arg6[1, 2, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_4 = tensor.extract_slice %arg6[1, 0, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_5 = tensor.extract_slice %arg6[1, 1, 2] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_6 = tensor.extract_slice %arg6[1, 1, 0] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_7 = tensor.extract_slice %arg5[1, 1, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_1, %extracted_slice_2, %extracted_slice_3, %extracted_slice_4, %extracted_slice_5, %extracted_slice_6 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%extracted_slice_7 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64):
        %20 = arith.mulf %in_17, %cst : f64
        %21 = arith.subf %in, %20 : f64
        %22 = arith.addf %21, %in_18 : f64
        %23 = arith.mulf %22, %cst_0 : f64
        %24 = arith.subf %in_19, %20 : f64
        %25 = arith.addf %24, %in_20 : f64
        %26 = arith.mulf %25, %cst_0 : f64
        %27 = arith.addf %23, %26 : f64
        %28 = arith.subf %in_21, %20 : f64
        %29 = arith.addf %28, %in_22 : f64
        %30 = arith.mulf %29, %cst_0 : f64
        %31 = arith.addf %27, %30 : f64
        %32 = arith.addf %31, %in_17 : f64
        linalg.yield %32 : f64
      } -> tensor<?x?x?xf64>
      %inserted_slice = tensor.insert_slice %15 into %arg5[1, 1, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<?x?x?xf64>
      %16 = arith.subi %11, %c1 : index
      %17 = affine.apply #map()[%2]
      %18 = arith.subi %17, %c1 : index
      %extracted_slice_8 = tensor.extract_slice %inserted_slice[2, 1, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_9 = tensor.extract_slice %inserted_slice[1, 1, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[0, 1, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_11 = tensor.extract_slice %inserted_slice[1, 2, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_12 = tensor.extract_slice %inserted_slice[1, 0, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_13 = tensor.extract_slice %inserted_slice[1, 1, 2] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_14 = tensor.extract_slice %inserted_slice[1, 1, 0] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %extracted_slice_15 = tensor.extract_slice %arg6[1, 1, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %19 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%extracted_slice_8, %extracted_slice_9, %extracted_slice_10, %extracted_slice_11, %extracted_slice_12, %extracted_slice_13, %extracted_slice_14 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%extracted_slice_15 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64):
        %20 = arith.mulf %in_17, %cst : f64
        %21 = arith.subf %in, %20 : f64
        %22 = arith.addf %21, %in_18 : f64
        %23 = arith.mulf %22, %cst_0 : f64
        %24 = arith.subf %in_19, %20 : f64
        %25 = arith.addf %24, %in_20 : f64
        %26 = arith.mulf %25, %cst_0 : f64
        %27 = arith.addf %23, %26 : f64
        %28 = arith.subf %in_21, %20 : f64
        %29 = arith.addf %28, %in_22 : f64
        %30 = arith.mulf %29, %cst_0 : f64
        %31 = arith.addf %27, %30 : f64
        %32 = arith.addf %31, %in_17 : f64
        linalg.yield %32 : f64
      } -> tensor<?x?x?xf64>
      %inserted_slice_16 = tensor.insert_slice %19 into %arg6[1, 1, 1] [%18, %16, %10] [1, 1, 1] : tensor<?x?x?xf64> into tensor<?x?x?xf64>
      affine.yield %inserted_slice, %inserted_slice_16 : tensor<?x?x?xf64>, tensor<?x?x?xf64>
    }
    %13 = bufferization.to_memref %12#1 : memref<?x?x?xf64>
    memref.copy %13, %arg2 : memref<?x?x?xf64> to memref<?x?x?xf64>
    %14 = bufferization.to_memref %12#0 : memref<?x?x?xf64>
    memref.copy %14, %arg3 : memref<?x?x?xf64> to memref<?x?x?xf64>
    return
  }
}

