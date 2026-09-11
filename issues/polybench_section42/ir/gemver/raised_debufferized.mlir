#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg6 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg7 restrict : memref<?xf64>
    %5 = bufferization.to_tensor %arg8 restrict : memref<?xf64>
    %6 = bufferization.to_tensor %arg9 restrict : memref<?xf64>
    %7 = bufferization.to_tensor %arg10 restrict : memref<?xf64>
    %8 = bufferization.to_tensor %arg11 restrict : memref<?xf64>
    %9 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %1[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %3[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_2 = tensor.extract_slice %4[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_3 = tensor.extract_slice %0[0, 0] [%9, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map, #map1, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_3 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_10: f64, %in_11: f64, %in_12: f64, %out: f64):
      %17 = arith.mulf %in, %in_10 : f64
      %18 = arith.addf %out, %17 : f64
      %19 = arith.mulf %in_11, %in_12 : f64
      %20 = arith.addf %18, %19 : f64
      linalg.yield %20 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %10 into %0[0, 0] [%9, %9] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %11 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %11, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %extracted_slice_4 = tensor.extract_slice %7[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_5 = tensor.extract_slice %6[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %12 = linalg.generic {doc = "", indexing_maps = [#map3, #map1, #map], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%10, %extracted_slice_4 : tensor<?x?xf64>, tensor<?xf64>) outs(%extracted_slice_5 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_10: f64, %out: f64):
      %17 = arith.mulf %arg2, %in : f64
      %18 = arith.mulf %17, %in_10 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    } -> tensor<?xf64>
    %inserted_slice_6 = tensor.insert_slice %12 into %6[0] [%9] [1] : tensor<?xf64> into tensor<?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map4, #map4], iterator_types = ["parallel"], library_call = ""} ins(%8 : tensor<?xf64>) outs(%inserted_slice_6 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %17 = arith.addf %out, %in : f64
      linalg.yield %17 : f64
    } -> tensor<?xf64>
    %14 = bufferization.to_memref %13 : memref<?xf64>
    memref.copy %14, %arg9 : memref<?xf64> to memref<?xf64>
    %extracted_slice_7 = tensor.extract_slice %13[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_8 = tensor.extract_slice %5[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map2, #map1, #map], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%10, %extracted_slice_7 : tensor<?x?xf64>, tensor<?xf64>) outs(%extracted_slice_8 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_10: f64, %out: f64):
      %17 = arith.mulf %arg1, %in : f64
      %18 = arith.mulf %17, %in_10 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    } -> tensor<?xf64>
    %inserted_slice_9 = tensor.insert_slice %15 into %5[0] [%9] [1] : tensor<?xf64> into tensor<?xf64>
    %16 = bufferization.to_memref %inserted_slice_9 : memref<?xf64>
    memref.copy %16, %arg8 : memref<?xf64> to memref<?xf64>
    return
  }
}

