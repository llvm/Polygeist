#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0) -> (d0, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @trsv_reads_b(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %5 = polygeist.submap(%2, %3) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %6 = polygeist.submap(%1, %3) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%5 : tensor<?xf64>) outs(%6 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<?xf64>
    %8 = polygeist.submapInverse(%1, %7, %3) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %9 = polygeist.submap(%0, %3, %4) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %10 = polygeist.submap(%2, %3, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %11 = polygeist.submap(%8, %3, %4) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %12 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%9, %10 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%11 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %19 = linalg.index 0 : index
      %20 = arith.mulf %in, %in_0 : f64
      %21 = arith.subf %out, %20 : f64
      %22 = linalg.index 1 : index
      %23 = arith.cmpi slt, %22, %19 : index
      %24 = arith.select %23, %21, %out : f64
      linalg.yield %24 : f64
    } -> tensor<?x?xf64>
    %13 = polygeist.submapInverse(%8, %12, %3, %4) {map = #map3} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %14 = polygeist.submap(%0, %3) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
    %15 = polygeist.submap(%13, %3) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %16 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%14 : tensor<?xf64>) outs(%15 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %19 = arith.divf %out, %in : f64
      linalg.yield %19 : f64
    } -> tensor<?xf64>
    %17 = polygeist.submapInverse(%13, %16, %3) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %18 = bufferization.to_memref %17 : memref<?xf64>
    memref.copy %18, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

