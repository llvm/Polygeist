#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x30xf64>, %arg4: memref<?x30xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg7 : memref<?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?xf64>
    %2 = bufferization.to_tensor %arg5 : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 : memref<?x30xf64>
    %4 = bufferization.to_tensor %arg3 : memref<?x30xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = polygeist.submap(%2, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%6 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    %8 = polygeist.submapInverse(%2, %7, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %9 = polygeist.submap(%0, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%9 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    %11 = polygeist.submapInverse(%0, %10, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %12 = polygeist.submap(%4, %5, %5) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
    %13 = polygeist.submap(%8, %5, %5) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %14 = polygeist.submap(%1, %5, %5) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%12, %14 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%13 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %28 = arith.mulf %in, %in_0 : f64
      %29 = arith.addf %28, %out : f64
      linalg.yield %29 : f64
    } -> tensor<?x?xf64>
    %16 = polygeist.submapInverse(%8, %15, %5, %5) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %17 = bufferization.to_memref %16 : memref<?xf64>
    memref.copy %17, %arg5 : memref<?xf64> to memref<?xf64>
    %18 = polygeist.submap(%3, %5, %5) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
    %19 = polygeist.submap(%1, %5, %5) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %20 = polygeist.submap(%11, %5, %5) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %21 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%18, %19 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%20 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %28 = arith.mulf %in, %in_0 : f64
      %29 = arith.addf %28, %out : f64
      linalg.yield %29 : f64
    } -> tensor<?x?xf64>
    %22 = polygeist.submapInverse(%11, %21, %5, %5) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %23 = polygeist.submap(%16, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %24 = polygeist.submap(%22, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%23 : tensor<?xf64>) outs(%24 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %28 = arith.mulf %arg1, %in : f64
      %29 = arith.mulf %arg2, %out : f64
      %30 = arith.addf %28, %29 : f64
      linalg.yield %30 : f64
    } -> tensor<?xf64>
    %26 = polygeist.submapInverse(%22, %25, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %27 = bufferization.to_memref %26 : memref<?xf64>
    memref.copy %27, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
}

