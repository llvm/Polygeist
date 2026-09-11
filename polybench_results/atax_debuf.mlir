#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0)[s0] -> (s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<?x42xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = bufferization.to_tensor %arg2 : memref<?x42xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = polygeist.submap(%1, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%5 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    %7 = polygeist.submapInverse(%1, %6, %4) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %8 = arith.index_cast %arg0 : i32 to index
    %9:2 = affine.for %arg6 = 0 to %8 iter_args(%arg7 = %7, %arg8 = %0) -> (tensor<?xf64>, tensor<?xf64>) {
      %inserted = tensor.insert %cst into %arg8[%arg6] : tensor<?xf64>
      %12 = polygeist.submap(%3, %arg6, %4) {map = #map1} : (tensor<?x42xf64>, index, index) -> tensor<?xf64>
      %13 = polygeist.submap(%2, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
      %14 = polygeist.submap(%inserted, %arg6, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%12, %13 : tensor<?xf64>, tensor<?xf64>) outs(%14 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %22 = arith.mulf %in, %in_0 : f64
        %23 = arith.addf %out, %22 : f64
        linalg.yield %23 : f64
      } -> tensor<?xf64>
      %16 = polygeist.submapInverse(%inserted, %15, %arg6, %4) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
      %17 = polygeist.submap(%3, %arg6, %4) {map = #map1} : (tensor<?x42xf64>, index, index) -> tensor<?xf64>
      %18 = polygeist.submap(%arg7, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
      %19 = polygeist.submap(%16, %arg6, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %20 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%17, %19 : tensor<?xf64>, tensor<?xf64>) outs(%18 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %22 = arith.mulf %in, %in_0 : f64
        %23 = arith.addf %out, %22 : f64
        linalg.yield %23 : f64
      } -> tensor<?xf64>
      %21 = polygeist.submapInverse(%arg7, %20, %4) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
      affine.yield %21, %16 : tensor<?xf64>, tensor<?xf64>
    }
    %10 = bufferization.to_memref %9#1 : memref<?xf64>
    memref.copy %10, %arg5 : memref<?xf64> to memref<?xf64>
    %11 = bufferization.to_memref %9#0 : memref<?xf64>
    memref.copy %11, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}

