#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_cholesky(%arg0: i32, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 restrict : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = affine.for %arg2 = 0 to %1 iter_args(%arg3 = %0) -> (tensor<?x?xf64>) {
      %5 = arith.subi %arg2, %c1 : index
      %6 = affine.for %arg4 = 0 to #map(%arg2) iter_args(%arg5 = %arg3) -> (tensor<?x?xf64>) {
        %extracted_slice_1 = tensor.extract_slice %arg5[%arg2, 0] [1, %5] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_2 = tensor.extract_slice %arg5[%arg4, 0] [1, %5] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_3 = tensor.extract_slice %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
        %9 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_1, %extracted_slice_2 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_3 : tensor<f64>) {
        ^bb0(%in: f64, %in_8: f64, %out: f64):
          %11 = arith.mulf %in, %in_8 : f64
          %12 = arith.subf %out, %11 : f64
          %13 = linalg.index 0 : index
          %14 = arith.cmpi slt, %13, %arg4 : index
          %15 = arith.select %14, %12, %out : f64
          linalg.yield %15 : f64
        } -> tensor<f64>
        %inserted_slice_4 = tensor.insert_slice %9 into %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<f64> into tensor<?x?xf64>
        %extracted_5 = tensor.extract %inserted_slice_4[%arg4, %arg4] : tensor<?x?xf64>
        %extracted_6 = tensor.extract %inserted_slice_4[%arg2, %arg4] : tensor<?x?xf64>
        %10 = arith.divf %extracted_6, %extracted_5 : f64
        %inserted_7 = tensor.insert %10 into %inserted_slice_4[%arg2, %arg4] : tensor<?x?xf64>
        affine.yield %inserted_7 : tensor<?x?xf64>
      }
      %extracted_slice = tensor.extract_slice %6[%arg2, 0] [1, %2] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_0 = tensor.extract_slice %6[%arg2, %arg2] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
      %7 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf64>) outs(%extracted_slice_0 : tensor<f64>) {
      ^bb0(%in: f64, %out: f64):
        %9 = arith.mulf %in, %in : f64
        %10 = arith.subf %out, %9 : f64
        %11 = linalg.index 0 : index
        %12 = arith.cmpi slt, %11, %arg2 : index
        %13 = arith.select %12, %10, %out : f64
        linalg.yield %13 : f64
      } -> tensor<f64>
      %inserted_slice = tensor.insert_slice %7 into %6[%arg2, %arg2] [1, 1] [1, 1] : tensor<f64> into tensor<?x?xf64>
      %extracted = tensor.extract %inserted_slice[%arg2, %arg2] : tensor<?x?xf64>
      %8 = math.sqrt %extracted : f64
      %inserted = tensor.insert %8 into %inserted_slice[%arg2, %arg2] : tensor<?x?xf64>
      affine.yield %inserted : tensor<?x?xf64>
    }
    %4 = bufferization.to_memref %3 : memref<?x?xf64>
    memref.copy %4, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

