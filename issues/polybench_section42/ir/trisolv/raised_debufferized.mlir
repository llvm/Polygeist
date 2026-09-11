#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %1) -> (tensor<?xf64>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf64>
      %inserted = tensor.insert %extracted into %arg5[%arg4] : tensor<?xf64>
      %extracted_slice = tensor.extract_slice %0[%arg4, 0] [1, %4] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_0 = tensor.extract_slice %inserted[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
      %extracted_slice_1 = tensor.extract_slice %inserted[%arg4] [1] [1] : tensor<?xf64> to tensor<f64>
      %7 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_1 : tensor<f64>) {
      ^bb0(%in: f64, %in_5: f64, %out: f64):
        %9 = arith.mulf %in, %in_5 : f64
        %10 = arith.subf %out, %9 : f64
        %11 = linalg.index 0 : index
        %12 = arith.cmpi slt, %11, %arg4 : index
        %13 = arith.select %12, %10, %out : f64
        linalg.yield %13 : f64
      } -> tensor<f64>
      %inserted_slice = tensor.insert_slice %7 into %inserted[%arg4] [1] [1] : tensor<f64> into tensor<?xf64>
      %extracted_2 = tensor.extract %inserted_slice[%arg4] : tensor<?xf64>
      %extracted_3 = tensor.extract %0[%arg4, %arg4] : tensor<?x?xf64>
      %8 = arith.divf %extracted_2, %extracted_3 : f64
      %inserted_4 = tensor.insert %8 into %inserted_slice[%arg4] : tensor<?xf64>
      affine.yield %inserted_4 : tensor<?xf64>
    }
    %6 = bufferization.to_memref %5 : memref<?xf64>
    memref.copy %6, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

