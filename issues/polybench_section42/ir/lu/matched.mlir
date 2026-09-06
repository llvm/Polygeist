#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_lu(%arg0: i32, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 restrict : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = affine.for %arg2 = 0 to %1 iter_args(%arg3 = %0) -> (tensor<?x?xf64>) {
      %5 = arith.subi %arg2, %c1 : index
      %6 = affine.for %arg4 = 0 to #map(%arg2) iter_args(%arg5 = %arg3) -> (tensor<?x?xf64>) {
        %extracted_slice_2 = tensor.extract_slice %arg5[%arg2, 0] [1, %5] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_3 = tensor.extract_slice %arg5[0, %arg4] [%5, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_4 = tensor.extract_slice %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
        %8 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_4 : tensor<f64>) {
        ^bb0(%in: f64, %in_7: f64, %out: f64):
          %10 = arith.mulf %in, %in_7 : f64
          %11 = arith.subf %out, %10 : f64
          %12 = linalg.index 0 : index
          %13 = arith.cmpi slt, %12, %arg4 : index
          %14 = arith.select %13, %11, %out : f64
          linalg.yield %14 : f64
        } -> tensor<f64>
        %inserted_slice_5 = tensor.insert_slice %8 into %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<f64> into tensor<?x?xf64>
        %extracted = tensor.extract %inserted_slice_5[%arg4, %arg4] : tensor<?x?xf64>
        %extracted_6 = tensor.extract %inserted_slice_5[%arg2, %arg4] : tensor<?x?xf64>
        %9 = arith.divf %extracted_6, %extracted : f64
        %inserted = tensor.insert %9 into %inserted_slice_5[%arg2, %arg4] : tensor<?x?xf64>
        affine.yield %inserted : tensor<?x?xf64>
      }
      %extracted_slice = tensor.extract_slice %6[%arg2, 0] [1, %2] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_0 = tensor.extract_slice %6[0, 0] [%2, %1] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_1 = tensor.extract_slice %6[%arg2, 0] [1, %1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %7 = linalg.generic {doc = "", indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?xf64>, tensor<?x?xf64>) outs(%extracted_slice_1 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %out: f64):
        %8 = arith.mulf %in, %in_2 : f64
        %9 = arith.subf %out, %8 : f64
        %10 = linalg.index 1 : index
        %11 = arith.cmpi slt, %10, %arg2 : index
        %12 = arith.select %11, %9, %out : f64
        linalg.yield %12 : f64
      } -> tensor<?xf64>
      %inserted_slice = tensor.insert_slice %7 into %6[%arg2, 0] [1, %1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      affine.yield %inserted_slice : tensor<?x?xf64>
    }
    %4 = bufferization.to_memref %3 : memref<?x?xf64>
    memref.copy %4, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

