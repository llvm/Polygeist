#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_mm(%arg0: memref<?x16xf64>, %arg1: memref<?x16xf64>, %arg2: memref<?x16xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x16xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?x16xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?x16xf64>
    %extracted_slice = tensor.extract_slice %2[0, 0] [%c16, %c16] [1, 1] : tensor<?x16xf64> to tensor<?x?xf64>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %0[0, 0] [%c16, %c16] [1, 1] : tensor<?x16xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %1[0, 0] [%c16, %c16] [1, 1] : tensor<?x16xf64> to tensor<?x?xf64>
    %4 = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%extracted_slice_0, %extracted_slice_1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%3 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %6 = arith.mulf %in, %in_2 : f64
      %7 = arith.addf %out, %6 : f64
      linalg.yield %7 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %4 into %2[0, 0] [%c16, %c16] [1, 1] : tensor<?x?xf64> into tensor<?x16xf64>
    %5 = bufferization.to_memref %inserted_slice : memref<?x16xf64>
    memref.copy %5, %arg2 : memref<?x16xf64> to memref<?x16xf64>
    return
  }
}

