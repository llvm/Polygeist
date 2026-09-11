#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x18xf64>, %arg7: memref<?x22xf64>, %arg8: memref<?x18xf64>, %arg9: memref<?x24xf64>, %arg10: memref<?x24xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg10 : memref<?x24xf64>
    %1 = bufferization.to_tensor %arg9 : memref<?x24xf64>
    %2 = bufferization.to_tensor %arg8 : memref<?x18xf64>
    %3 = bufferization.to_tensor %arg7 : memref<?x22xf64>
    %4 = bufferization.to_tensor %arg6 : memref<?x18xf64>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg0 : i32 to index
    %9 = polygeist.submap(%4, %7, %8) {map = #map} : (tensor<?x18xf64>, index, index) -> tensor<?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%9 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %11 = polygeist.submapInverse(%4, %10, %7, %8) {map = #map} : (tensor<?x18xf64>, tensor<?x?xf64>, index, index) -> tensor<?x18xf64>
    %12 = polygeist.submap(%11, %5, %7, %8) {map = #map2} : (tensor<?x18xf64>, index, index, index) -> tensor<?x?x?xf64>
    %13 = polygeist.submap(%3, %5, %7, %8) {map = #map3} : (tensor<?x22xf64>, index, index, index) -> tensor<?x?x?xf64>
    %14 = polygeist.submap(%2, %5, %7, %8) {map = #map4} : (tensor<?x18xf64>, index, index, index) -> tensor<?x?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%13, %14 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%12 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %27 = arith.mulf %arg4, %in : f64
      %28 = arith.mulf %27, %in_0 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    } -> tensor<?x?x?xf64>
    %16 = polygeist.submapInverse(%11, %15, %5, %7, %8) {map = #map2} : (tensor<?x18xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x18xf64>
    %17 = bufferization.to_memref %16 : memref<?x18xf64>
    memref.copy %17, %arg6 : memref<?x18xf64> to memref<?x18xf64>
    %18 = polygeist.submap(%0, %6, %8) {map = #map} : (tensor<?x24xf64>, index, index) -> tensor<?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%18 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %27 = arith.mulf %out, %arg5 : f64
      linalg.yield %27 : f64
    } -> tensor<?x?xf64>
    %20 = polygeist.submapInverse(%0, %19, %6, %8) {map = #map} : (tensor<?x24xf64>, tensor<?x?xf64>, index, index) -> tensor<?x24xf64>
    %21 = polygeist.submap(%16, %7, %6, %8) {map = #map3} : (tensor<?x18xf64>, index, index, index) -> tensor<?x?x?xf64>
    %22 = polygeist.submap(%1, %7, %6, %8) {map = #map4} : (tensor<?x24xf64>, index, index, index) -> tensor<?x?x?xf64>
    %23 = polygeist.submap(%20, %7, %6, %8) {map = #map2} : (tensor<?x24xf64>, index, index, index) -> tensor<?x?x?xf64>
    %24 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%21, %22 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%23 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %27 = arith.mulf %in, %in_0 : f64
      %28 = arith.addf %out, %27 : f64
      linalg.yield %28 : f64
    } -> tensor<?x?x?xf64>
    %25 = polygeist.submapInverse(%20, %24, %7, %6, %8) {map = #map2} : (tensor<?x24xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x24xf64>
    %26 = bufferization.to_memref %25 : memref<?x24xf64>
    memref.copy %26, %arg10 : memref<?x24xf64> to memref<?x24xf64>
    return
  }
}

