#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.69999999999999996 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg6 restrict : memref<?xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.subi %4, %c1 : index
    %8 = arith.subi %5, %c1 : index
    %9 = affine.apply #map()[%5]
    %10 = affine.apply #map()[%4]
    %11:3 = affine.for %arg7 = 0 to %6 iter_args(%arg8 = %1, %arg9 = %0, %arg10 = %2) -> (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) {
      %extracted_slice = tensor.extract_slice %3[%arg7] [1] [1] : tensor<?xf64> to tensor<f64>
      %extracted_slice_1 = tensor.extract_slice %arg8[0, 0] [1, %5] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice : tensor<f64>) outs(%extracted_slice_1 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %inserted_slice = tensor.insert_slice %15 into %arg8[0, 0] [1, %5] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %extracted_slice_2 = tensor.extract_slice %arg10[1, 0] [%7, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_3 = tensor.extract_slice %arg10[0, 0] [%7, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_4 = tensor.extract_slice %inserted_slice[1, 0] [%7, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %16 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_4 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_16: f64, %out: f64):
        %19 = arith.subf %in, %in_16 : f64
        %20 = arith.mulf %19, %cst_0 : f64
        %21 = arith.subf %out, %20 : f64
        linalg.yield %21 : f64
      } -> tensor<?x?xf64>
      %inserted_slice_5 = tensor.insert_slice %16 into %inserted_slice[1, 0] [%7, %5] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %extracted_slice_6 = tensor.extract_slice %arg10[0, 1] [%4, %8] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_7 = tensor.extract_slice %arg10[0, 0] [%4, %8] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_8 = tensor.extract_slice %arg9[0, 1] [%4, %8] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_6, %extracted_slice_7 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_8 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_16: f64, %out: f64):
        %19 = arith.subf %in, %in_16 : f64
        %20 = arith.mulf %19, %cst_0 : f64
        %21 = arith.subf %out, %20 : f64
        linalg.yield %21 : f64
      } -> tensor<?x?xf64>
      %inserted_slice_9 = tensor.insert_slice %17 into %arg9[0, 1] [%4, %8] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice_9[0, 1] [%10, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_11 = tensor.extract_slice %inserted_slice_9[0, 0] [%10, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_12 = tensor.extract_slice %inserted_slice_5[1, 0] [%10, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_13 = tensor.extract_slice %inserted_slice_5[0, 0] [%10, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_14 = tensor.extract_slice %arg10[0, 0] [%10, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %18 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_10, %extracted_slice_11, %extracted_slice_12, %extracted_slice_13 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_14 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_16: f64, %in_17: f64, %in_18: f64, %out: f64):
        %19 = arith.subf %in, %in_16 : f64
        %20 = arith.addf %19, %in_17 : f64
        %21 = arith.subf %20, %in_18 : f64
        %22 = arith.mulf %21, %cst : f64
        %23 = arith.subf %out, %22 : f64
        linalg.yield %23 : f64
      } -> tensor<?x?xf64>
      %inserted_slice_15 = tensor.insert_slice %18 into %arg10[0, 0] [%10, %9] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      affine.yield %inserted_slice_5, %inserted_slice_9, %inserted_slice_15 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>
    }
    %12 = bufferization.to_memref %11#2 : memref<?x?xf64>
    memref.copy %12, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %13 = bufferization.to_memref %11#1 : memref<?x?xf64>
    memref.copy %13, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %14 = bufferization.to_memref %11#0 : memref<?x?xf64>
    memref.copy %14, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

