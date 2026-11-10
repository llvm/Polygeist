#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map4 = affine_map<(d0, d1) -> (d0 + 1, d1 + 2)>
#map5 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
#map6 = affine_map<(d0, d1) -> (d0, d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.apply #map()[%2]
    %5 = affine.apply #map()[%2]
    %6 = arith.subi %5, %c1 : index
    %7 = affine.apply #map()[%2]
    %8 = arith.subi %7, %c1 : index
    %9 = arith.subi %4, %c1 : index
    %10 = affine.apply #map()[%2]
    %11 = arith.subi %10, %c1 : index
    %12:2 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x?xf64>, tensor<?x?xf64>) {
      %15 = tensor.empty(%8, %6) : tensor<?x?xf64>
      %16 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg5 : tensor<?x?xf64>) outs(%15 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?x?xf64>
      %dim = tensor.dim %arg6, %c0 : tensor<?x?xf64>
      %extracted_slice = tensor.extract_slice %arg6[0, 0] [%dim, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %dim_0 = tensor.dim %arg6, %c1 : tensor<?x?xf64>
      %extracted_slice_1 = tensor.extract_slice %arg6[0, 0] [%8, %dim_0] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map1, #map3, #map4, #map5, #map6, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%arg6, %extracted_slice, %arg6, %arg6, %extracted_slice_1 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%16 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %out: f64):
        %21 = arith.addf %in, %in_7 : f64
        %22 = arith.addf %21, %in_8 : f64
        %23 = arith.addf %22, %in_9 : f64
        %24 = arith.addf %23, %in_10 : f64
        %25 = arith.mulf %24, %cst : f64
        linalg.yield %25 : f64
      } -> tensor<?x?xf64>
      %inserted_slice = tensor.insert_slice %17 into %arg5[1, 1] [%8, %6] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %18 = tensor.empty(%11, %9) : tensor<?x?xf64>
      %19 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<?x?xf64>) outs(%18 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?x?xf64>
      %dim_2 = tensor.dim %arg5, %c0 : tensor<?x?xf64>
      %extracted_slice_3 = tensor.extract_slice %inserted_slice[0, 0] [%dim_2, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %dim_4 = tensor.dim %arg5, %c1 : tensor<?x?xf64>
      %extracted_slice_5 = tensor.extract_slice %inserted_slice[0, 0] [%11, %dim_4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %20 = linalg.generic {doc = "", indexing_maps = [#map1, #map3, #map4, #map5, #map6, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%inserted_slice, %extracted_slice_3, %inserted_slice, %inserted_slice, %extracted_slice_5 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%19 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %out: f64):
        %21 = arith.addf %in, %in_7 : f64
        %22 = arith.addf %21, %in_8 : f64
        %23 = arith.addf %22, %in_9 : f64
        %24 = arith.addf %23, %in_10 : f64
        %25 = arith.mulf %24, %cst : f64
        linalg.yield %25 : f64
      } -> tensor<?x?xf64>
      %inserted_slice_6 = tensor.insert_slice %20 into %arg6[1, 1] [%11, %9] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      affine.yield %inserted_slice, %inserted_slice_6 : tensor<?x?xf64>, tensor<?x?xf64>
    }
    %13 = bufferization.to_memref %12#0 : memref<?x?xf64>
    memref.copy %13, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %14 = bufferization.to_memref %12#1 : memref<?x?xf64>
    memref.copy %14, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

