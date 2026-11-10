#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @rank2(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 : memref<?xf64>
    %4 = bufferization.to_tensor %arg5 : memref<?xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = polygeist.submap(%1, %5, %5) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %7 = polygeist.submap(%2, %5, %5) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %8 = polygeist.submap(%3, %5, %5) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %9 = polygeist.submap(%4, %5, %5) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %10 = polygeist.submap(%0, %5, %5) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %11 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%6, %7, %8, %9 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%10 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %14 = arith.mulf %in, %in_0 : f64
      %15 = arith.addf %out, %14 : f64
      %16 = arith.mulf %in_1, %in_2 : f64
      %17 = arith.addf %15, %16 : f64
      linalg.yield %17 : f64
    } -> tensor<?x?xf64>
    %12 = polygeist.submapInverse(%0, %11, %5, %5) {map = #map2} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %13 = bufferization.to_memref %12 : memref<?x?xf64>
    memref.copy %13, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

