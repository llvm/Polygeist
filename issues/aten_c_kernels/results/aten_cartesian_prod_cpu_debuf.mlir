#map = affine_map<(d0, d1) -> (d1 + d0 * 12, 0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1 + d0 * 12, 1)>
#map4 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cartesian_prod_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?x2xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x2xf32>
    %extracted_slice = tensor.extract_slice %0[0] [%c16] [1] : tensor<?xf32> to tensor<?xf32>
    %3 = polygeist.submap(%2, %c16, %c12) {map = #map} : (tensor<?x2xf32>, index, index) -> tensor<?x?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%3 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    %5 = polygeist.submapInverse(%2, %4, %c16, %c12) {map = #map} : (tensor<?x2xf32>, tensor<?x?xf32>, index, index) -> tensor<?x2xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0] [%c12] [1] : tensor<?xf32> to tensor<?xf32>
    %6 = polygeist.submap(%5, %c16, %c12) {map = #map3} : (tensor<?x2xf32>, index, index) -> tensor<?x?xf32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_0 : tensor<?xf32>) outs(%6 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    %8 = polygeist.submapInverse(%5, %7, %c16, %c12) {map = #map3} : (tensor<?x2xf32>, tensor<?x?xf32>, index, index) -> tensor<?x2xf32>
    %9 = bufferization.to_memref %8 : memref<?x2xf32>
    memref.copy %9, %arg2 : memref<?x2xf32> to memref<?x2xf32>
    return
  }
}

