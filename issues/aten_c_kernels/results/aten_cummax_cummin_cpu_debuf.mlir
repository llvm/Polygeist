#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cummax_cummin_cpu(%arg0: memref<?x64xf32>, %arg1: i32, %arg2: memref<?x64xf32>, %arg3: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?x64xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?x64xi32>
    %3 = arith.cmpi ne, %arg1, %c0_i32 : i32
    %4 = tensor.empty(%c16) : tensor<?xi32>
    %5 = tensor.empty(%c16) : tensor<?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%4 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c16, 1] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %5[0] [%c16] [1] : tensor<?xf32> to tensor<?xf32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_0 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?xf32>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_2 = tensor.extract_slice %0[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %0[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_4 = tensor.extract_slice %1[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_5 = tensor.extract_slice %2[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xi32> to tensor<?x?xi32>
    %extracted_slice_6 = tensor.extract_slice %6[0] [%c16] [1] : tensor<?xi32> to tensor<?xi32>
    %8:4 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map2, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice_1, %extracted_slice_2, %extracted_slice_3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_4, %extracted_slice_5, %extracted_slice_6, %7 : tensor<?x?xf32>, tensor<?x?xi32>, tensor<?xi32>, tensor<?xf32>) {
    ^bb0(%in: f32, %in_8: f32, %in_9: f32, %out: f32, %out_10: i32, %out_11: i32, %out_12: f32):
      %11 = linalg.index 1 : index
      %12 = arith.index_cast %11 : index to i32
      %13 = arith.cmpf oge, %in, %out_12 : f32
      %14 = arith.select %3, %13, %false : i1
      %15 = arith.cmpf ole, %in_8, %out_12 : f32
      %16 = arith.select %3, %false, %15 : i1
      %17 = arith.select %14, %true, %16 : i1
      %18 = arith.select %17, %12, %out_11 : i32
      %19 = arith.select %17, %in_9, %out_12 : f32
      linalg.yield %19, %18, %18, %19 : f32, i32, i32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xi32>, tensor<?xi32>, tensor<?xf32>)
    %inserted_slice = tensor.insert_slice %8#0 into %1[0, 0] [%c16, %c64] [1, 1] : tensor<?x?xf32> into tensor<?x64xf32>
    %9 = bufferization.to_memref %inserted_slice : memref<?x64xf32>
    memref.copy %9, %arg2 : memref<?x64xf32> to memref<?x64xf32>
    %inserted_slice_7 = tensor.insert_slice %8#1 into %2[0, 0] [%c16, %c64] [1, 1] : tensor<?x?xi32> into tensor<?x64xi32>
    %10 = bufferization.to_memref %inserted_slice_7 : memref<?x64xi32>
    memref.copy %10, %arg3 : memref<?x64xi32> to memref<?x64xi32>
    return
  }
}

