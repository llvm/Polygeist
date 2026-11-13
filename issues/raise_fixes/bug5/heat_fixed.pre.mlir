#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 1, d2 + 1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0 + 2, d1 + 1, d2 + 1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1 + 1, d2 + 1)>
#map5 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 2, d2 + 1)>
#map6 = affine_map<(d0, d1, d2) -> (d0 + 1, d1, d2 + 1)>
#map7 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 1, d2 + 2)>
#map8 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<?x?x?xf64>, %arg3: memref<?x?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
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
    %12:2 = affine.for %arg4 = 1 to 21 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
      %15 = tensor.empty(%9, %7, %5) : tensor<?x?x?xf64>
      %16 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<?x?x?xf64>) outs(%15 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?x?x?xf64>
      %dim = tensor.dim %arg6, %c1 : tensor<?x?x?xf64>
      %dim_1 = tensor.dim %arg6, %c2 : tensor<?x?x?xf64>
      %extracted_slice = tensor.extract_slice %arg6[0, 0, 0] [%9, %dim, %dim_1] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %dim_2 = tensor.dim %arg6, %c0 : tensor<?x?x?xf64>
      %dim_3 = tensor.dim %arg6, %c2 : tensor<?x?x?xf64>
      %extracted_slice_4 = tensor.extract_slice %arg6[0, 0, 0] [%dim_2, %7, %dim_3] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %dim_5 = tensor.dim %arg6, %c0 : tensor<?x?x?xf64>
      %dim_6 = tensor.dim %arg6, %c1 : tensor<?x?x?xf64>
      %extracted_slice_7 = tensor.extract_slice %arg6[0, 0, 0] [%dim_5, %dim_6, %5] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map3, #map1, #map4, #map5, #map6, #map7, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%arg6, %arg6, %extracted_slice, %arg6, %extracted_slice_4, %arg6, %extracted_slice_7 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%16 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %in_23: f64, %out: f64):
        %24 = arith.mulf %in_18, %cst : f64
        %25 = arith.subf %in, %24 : f64
        %26 = arith.addf %25, %in_19 : f64
        %27 = arith.mulf %26, %cst_0 : f64
        %28 = arith.subf %in_20, %24 : f64
        %29 = arith.addf %28, %in_21 : f64
        %30 = arith.mulf %29, %cst_0 : f64
        %31 = arith.addf %27, %30 : f64
        %32 = arith.subf %in_22, %24 : f64
        %33 = arith.addf %32, %in_23 : f64
        %34 = arith.mulf %33, %cst_0 : f64
        %35 = arith.addf %31, %34 : f64
        %36 = arith.addf %35, %in_18 : f64
        linalg.yield %36 : f64
      } -> tensor<?x?x?xf64>
      %inserted_slice = tensor.insert_slice %17 into %arg5[1, 1, 1] [%9, %7, %5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<?x?x?xf64>
      %18 = arith.subi %11, %c1 : index
      %19 = affine.apply #map()[%2]
      %20 = arith.subi %19, %c1 : index
      %21 = tensor.empty(%20, %18, %10) : tensor<?x?x?xf64>
      %22 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg6 : tensor<?x?x?xf64>) outs(%21 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?x?x?xf64>
      %dim_8 = tensor.dim %arg5, %c1 : tensor<?x?x?xf64>
      %dim_9 = tensor.dim %arg5, %c2 : tensor<?x?x?xf64>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[0, 0, 0] [%20, %dim_8, %dim_9] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %dim_11 = tensor.dim %arg5, %c0 : tensor<?x?x?xf64>
      %dim_12 = tensor.dim %arg5, %c2 : tensor<?x?x?xf64>
      %extracted_slice_13 = tensor.extract_slice %inserted_slice[0, 0, 0] [%dim_11, %18, %dim_12] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %dim_14 = tensor.dim %arg5, %c0 : tensor<?x?x?xf64>
      %dim_15 = tensor.dim %arg5, %c1 : tensor<?x?x?xf64>
      %extracted_slice_16 = tensor.extract_slice %inserted_slice[0, 0, 0] [%dim_14, %dim_15, %10] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?x?x?xf64>
      %23 = linalg.generic {doc = "", indexing_maps = [#map3, #map1, #map4, #map5, #map6, #map7, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%inserted_slice, %inserted_slice, %extracted_slice_10, %inserted_slice, %extracted_slice_13, %inserted_slice, %extracted_slice_16 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%22 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %in_23: f64, %out: f64):
        %24 = arith.mulf %in_18, %cst : f64
        %25 = arith.subf %in, %24 : f64
        %26 = arith.addf %25, %in_19 : f64
        %27 = arith.mulf %26, %cst_0 : f64
        %28 = arith.subf %in_20, %24 : f64
        %29 = arith.addf %28, %in_21 : f64
        %30 = arith.mulf %29, %cst_0 : f64
        %31 = arith.addf %27, %30 : f64
        %32 = arith.subf %in_22, %24 : f64
        %33 = arith.addf %32, %in_23 : f64
        %34 = arith.mulf %33, %cst_0 : f64
        %35 = arith.addf %31, %34 : f64
        %36 = arith.addf %35, %in_18 : f64
        linalg.yield %36 : f64
      } -> tensor<?x?x?xf64>
      %inserted_slice_17 = tensor.insert_slice %23 into %arg6[1, 1, 1] [%20, %18, %10] [1, 1, 1] : tensor<?x?x?xf64> into tensor<?x?x?xf64>
      affine.yield %inserted_slice, %inserted_slice_17 : tensor<?x?x?xf64>, tensor<?x?x?xf64>
    }
    %13 = bufferization.to_memref %12#1 : memref<?x?x?xf64>
    memref.copy %13, %arg2 : memref<?x?x?xf64> to memref<?x?x?xf64>
    %14 = bufferization.to_memref %12#0 : memref<?x?x?xf64>
    memref.copy %14, %arg3 : memref<?x?x?xf64> to memref<?x?x?xf64>
    return
  }
}

