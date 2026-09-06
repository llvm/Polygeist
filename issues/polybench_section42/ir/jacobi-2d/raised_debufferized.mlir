#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
      %extracted_slice = tensor.extract_slice %arg6[1, 1] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_0 = tensor.extract_slice %arg6[1, 0] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_1 = tensor.extract_slice %arg6[1, 2] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_2 = tensor.extract_slice %arg6[2, 1] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_3 = tensor.extract_slice %arg6[0, 1] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_4 = tensor.extract_slice %arg5[1, 1] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2, %extracted_slice_3 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_4 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %out: f64):
        %17 = arith.addf %in, %in_12 : f64
        %18 = arith.addf %17, %in_13 : f64
        %19 = arith.addf %18, %in_14 : f64
        %20 = arith.addf %19, %in_15 : f64
        %21 = arith.mulf %20, %cst : f64
        linalg.yield %21 : f64
      } -> tensor<?x?xf64>
      %inserted_slice = tensor.insert_slice %15 into %arg5[1, 1] [%8, %6] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %extracted_slice_5 = tensor.extract_slice %inserted_slice[1, 1] [%11, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_6 = tensor.extract_slice %inserted_slice[1, 0] [%11, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_7 = tensor.extract_slice %inserted_slice[1, 2] [%11, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_8 = tensor.extract_slice %inserted_slice[2, 1] [%11, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_9 = tensor.extract_slice %inserted_slice[0, 1] [%11, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_10 = tensor.extract_slice %arg6[1, 1] [%11, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %16 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_5, %extracted_slice_6, %extracted_slice_7, %extracted_slice_8, %extracted_slice_9 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_10 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %out: f64):
        %17 = arith.addf %in, %in_12 : f64
        %18 = arith.addf %17, %in_13 : f64
        %19 = arith.addf %18, %in_14 : f64
        %20 = arith.addf %19, %in_15 : f64
        %21 = arith.mulf %20, %cst : f64
        linalg.yield %21 : f64
      } -> tensor<?x?xf64>
      %inserted_slice_11 = tensor.insert_slice %16 into %arg6[1, 1] [%11, %9] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      affine.yield %inserted_slice, %inserted_slice_11 : tensor<?x?xf64>, tensor<?x?xf64>
    }
    %13 = bufferization.to_memref %12#1 : memref<?x?xf64>
    memref.copy %13, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    %14 = bufferization.to_memref %12#0 : memref<?x?xf64>
    memref.copy %14, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

