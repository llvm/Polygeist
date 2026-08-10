#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_floyd_warshall(%arg0: i32, %arg1: memref<?x60xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg1 : memref<?x60xi32>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = polygeist.submap(%0, %1, %1, %1) {map = #map} : (tensor<?x60xi32>, index, index, index) -> tensor<?x?x?xi32>
    %3 = polygeist.submap(%0, %1, %1, %1) {map = #map1} : (tensor<?x60xi32>, index, index, index) -> tensor<?x?x?xi32>
    %4 = polygeist.submap(%0, %1, %1, %1) {map = #map2} : (tensor<?x60xi32>, index, index, index) -> tensor<?x?x?xi32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["reduction", "parallel", "parallel"], library_call = ""} ins(%2, %3 : tensor<?x?x?xi32>, tensor<?x?x?xi32>) outs(%4 : tensor<?x?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.addi %in, %in_0 : i32
      %9 = arith.cmpi slt, %out, %8 : i32
      %10 = arith.select %9, %out, %8 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?x?xi32>
    %6 = polygeist.submapInverse(%0, %5, %1, %1, %1) {map = #map2} : (tensor<?x60xi32>, tensor<?x?x?xi32>, index, index, index) -> tensor<?x60xi32>
    %7 = bufferization.to_memref %6 : memref<?x60xi32>
    memref.copy %7, %arg1 : memref<?x60xi32> to memref<?x60xi32>
    return
  }
}

