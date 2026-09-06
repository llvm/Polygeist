#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_mvt(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg1 restrict : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %4[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?x?xf64>, tensor<?xf64>) outs(%extracted_slice_1 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_6: f64, %out: f64):
      %10 = arith.mulf %in, %in_6 : f64
      %11 = arith.addf %out, %10 : f64
      linalg.yield %11 : f64
    } -> tensor<?xf64>
    %inserted_slice = tensor.insert_slice %6 into %0[0] [%5] [1] : tensor<?xf64> into tensor<?xf64>
    %7 = bufferization.to_memref %inserted_slice : memref<?xf64>
    memref.copy %7, %arg1 : memref<?xf64> to memref<?xf64>
    %extracted_slice_2 = tensor.extract_slice %4[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_4 = tensor.extract_slice %1[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map3, #map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?x?xf64>, tensor<?xf64>) outs(%extracted_slice_4 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_6: f64, %out: f64):
      %10 = arith.mulf %in, %in_6 : f64
      %11 = arith.addf %out, %10 : f64
      linalg.yield %11 : f64
    } -> tensor<?xf64>
    %inserted_slice_5 = tensor.insert_slice %8 into %1[0] [%5] [1] : tensor<?xf64> into tensor<?xf64>
    %9 = bufferization.to_memref %inserted_slice_5 : memref<?xf64>
    memref.copy %9, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

