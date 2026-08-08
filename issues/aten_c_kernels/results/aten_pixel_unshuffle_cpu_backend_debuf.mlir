#map = affine_map<(d0, d1, d2, d3, d4) -> (0, d0, d3 + d1 * 2, d4 + d2 * 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (0, d4 + d3 * 2 + d0 * 4, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_pixel_unshuffle_cpu_backend(%arg0: memref<?x3x16x16xf32>, %arg1: memref<?x12x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c3 = arith.constant 3 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x3x16x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x12x8x8xf32>
    %2 = polygeist.submap(%0, %c3, %c8, %c8, %c2, %c2) {map = #map} : (tensor<?x3x16x16xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %3 = polygeist.submap(%1, %c3, %c8, %c8, %c2, %c2) {map = #map1} : (tensor<?x12x8x8xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%2 : tensor<?x?x?x?x?xf32>) outs(%3 : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?x?xf32>
    %5 = polygeist.submapInverse(%1, %4, %c3, %c8, %c8, %c2, %c2) {map = #map1} : (tensor<?x12x8x8xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?x12x8x8xf32>
    %6 = bufferization.to_memref %5 : memref<?x12x8x8xf32>
    memref.copy %6, %arg1 : memref<?x12x8x8xf32> to memref<?x12x8x8xf32>
    return
  }
}

