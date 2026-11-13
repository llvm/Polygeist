#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.for %arg5 = 0 to %3 iter_args(%arg6 = %1) -> (tensor<?x?xf64>) {
      %extracted_slice = tensor.extract_slice %0[0, %arg5] [%3, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_0 = tensor.extract_slice %arg6[0, 0] [%3, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg5, 0] [1, %2] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?xf64>, tensor<?x?xf64>) outs(%extracted_slice_1 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %out: f64):
        %8 = arith.mulf %in, %in_2 : f64
        %9 = arith.addf %out, %8 : f64
        %10 = linalg.index 1 : index
        %11 = affine.apply #map3(%arg5)
        %12 = arith.cmpi sge, %10, %11 : index
        %13 = arith.select %12, %9, %out : f64
        linalg.yield %13 : f64
      } -> tensor<?xf64>
      %7 = linalg.generic {doc = "", indexing_maps = [#map4], iterator_types = ["parallel"], library_call = ""} outs(%6 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %8 = arith.mulf %arg2, %out : f64
        linalg.yield %8 : f64
      } -> tensor<?xf64>
      %inserted_slice = tensor.insert_slice %7 into %arg6[%arg5, 0] [1, %2] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      affine.yield %inserted_slice : tensor<?x?xf64>
    }
    %5 = bufferization.to_memref %4 : memref<?x?xf64>
    memref.copy %5, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

