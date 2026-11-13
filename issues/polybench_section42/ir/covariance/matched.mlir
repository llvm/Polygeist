#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = kernel.launch @memset_zero_1D(%2) : (tensor<?xf64>) -> tensor<?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%0 : tensor<?x?xf64>) outs(%5 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %14 = arith.addf %out, %in : f64
      linalg.yield %14 : f64
    } -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%6 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %14 = arith.divf %out, %arg2 : f64
      linalg.yield %14 : f64
    } -> tensor<?xf64>
    %8 = bufferization.to_memref %7 : memref<?xf64>
    memref.copy %8, %arg5 : memref<?xf64> to memref<?xf64>
    %extracted_slice = tensor.extract_slice %7[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0] [%3, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %9 = linalg.generic {doc = "", indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice : tensor<?xf64>) outs(%extracted_slice_1 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %14 = arith.subf %out, %in : f64
      linalg.yield %14 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %9 into %0[0, 0] [%3, %4] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %10 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %10, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %11 = arith.subf %arg2, %cst : f64
    %12 = affine.for %arg6 = 0 to %4 iter_args(%arg7 = %1) -> (tensor<?x?xf64>) {
      %14 = affine.for %arg8 = #map(%arg6) to %4 iter_args(%arg9 = %arg7) -> (tensor<?x?xf64>) {
        %inserted = tensor.insert %cst_0 into %arg9[%arg6, %arg8] : tensor<?x?xf64>
        %extracted_slice_2 = tensor.extract_slice %inserted_slice[0, %arg6] [%3, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_3 = tensor.extract_slice %inserted_slice[0, %arg8] [%3, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_4 = tensor.extract_slice %inserted[%arg6, %arg8] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
        %15 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map5], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_4 : tensor<f64>) {
        ^bb0(%in: f64, %in_8: f64, %out: f64):
          %17 = arith.mulf %in, %in_8 : f64
          %18 = arith.addf %out, %17 : f64
          linalg.yield %18 : f64
        } -> tensor<f64>
        %inserted_slice_5 = tensor.insert_slice %15 into %inserted[%arg6, %arg8] [1, 1] [1, 1] : tensor<f64> into tensor<?x?xf64>
        %extracted = tensor.extract %inserted_slice_5[%arg6, %arg8] : tensor<?x?xf64>
        %16 = arith.divf %extracted, %11 : f64
        %inserted_6 = tensor.insert %16 into %inserted_slice_5[%arg6, %arg8] : tensor<?x?xf64>
        %inserted_7 = tensor.insert %16 into %inserted_6[%arg8, %arg6] : tensor<?x?xf64>
        affine.yield %inserted_7 : tensor<?x?xf64>
      }
      affine.yield %14 : tensor<?x?xf64>
    }
    %13 = bufferization.to_memref %12 : memref<?x?xf64>
    memref.copy %13, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

