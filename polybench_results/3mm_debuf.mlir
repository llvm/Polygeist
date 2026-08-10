#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x18xf64>, %arg6: memref<?x20xf64>, %arg7: memref<?x18xf64>, %arg8: memref<?x22xf64>, %arg9: memref<?x24xf64>, %arg10: memref<?x22xf64>, %arg11: memref<?x22xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg11 : memref<?x22xf64>
    %1 = bufferization.to_tensor %arg10 : memref<?x22xf64>
    %2 = bufferization.to_tensor %arg9 : memref<?x24xf64>
    %3 = bufferization.to_tensor %arg8 : memref<?x22xf64>
    %4 = bufferization.to_tensor %arg7 : memref<?x18xf64>
    %5 = bufferization.to_tensor %arg6 : memref<?x20xf64>
    %6 = bufferization.to_tensor %arg5 : memref<?x18xf64>
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg4 : i32 to index
    %10 = arith.index_cast %arg3 : i32 to index
    %11 = arith.index_cast %arg0 : i32 to index
    %12 = polygeist.submap(%6, %7, %11) {map = #map} : (tensor<?x18xf64>, index, index) -> tensor<?x?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%12 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %14 = polygeist.submapInverse(%6, %13, %7, %11) {map = #map} : (tensor<?x18xf64>, tensor<?x?xf64>, index, index) -> tensor<?x18xf64>
    %15 = polygeist.submap(%14, %8, %7, %11) {map = #map2} : (tensor<?x18xf64>, index, index, index) -> tensor<?x?x?xf64>
    %16 = polygeist.submap(%5, %8, %7, %11) {map = #map3} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %17 = polygeist.submap(%4, %8, %7, %11) {map = #map4} : (tensor<?x18xf64>, index, index, index) -> tensor<?x?x?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%16, %17 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%15 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %39 = arith.mulf %in, %in_0 : f64
      %40 = arith.addf %out, %39 : f64
      linalg.yield %40 : f64
    } -> tensor<?x?x?xf64>
    %19 = polygeist.submapInverse(%14, %18, %8, %7, %11) {map = #map2} : (tensor<?x18xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x18xf64>
    %20 = bufferization.to_memref %19 : memref<?x18xf64>
    memref.copy %20, %arg5 : memref<?x18xf64> to memref<?x18xf64>
    %21 = polygeist.submap(%3, %10, %7) {map = #map} : (tensor<?x22xf64>, index, index) -> tensor<?x?xf64>
    %22 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%21 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %23 = polygeist.submapInverse(%3, %22, %10, %7) {map = #map} : (tensor<?x22xf64>, tensor<?x?xf64>, index, index) -> tensor<?x22xf64>
    %24 = polygeist.submap(%23, %9, %10, %7) {map = #map2} : (tensor<?x22xf64>, index, index, index) -> tensor<?x?x?xf64>
    %25 = polygeist.submap(%2, %9, %10, %7) {map = #map3} : (tensor<?x24xf64>, index, index, index) -> tensor<?x?x?xf64>
    %26 = polygeist.submap(%1, %9, %10, %7) {map = #map4} : (tensor<?x22xf64>, index, index, index) -> tensor<?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%25, %26 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%24 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %39 = arith.mulf %in, %in_0 : f64
      %40 = arith.addf %out, %39 : f64
      linalg.yield %40 : f64
    } -> tensor<?x?x?xf64>
    %28 = polygeist.submapInverse(%23, %27, %9, %10, %7) {map = #map2} : (tensor<?x22xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x22xf64>
    %29 = bufferization.to_memref %28 : memref<?x22xf64>
    memref.copy %29, %arg8 : memref<?x22xf64> to memref<?x22xf64>
    %30 = polygeist.submap(%0, %10, %11) {map = #map} : (tensor<?x22xf64>, index, index) -> tensor<?x?xf64>
    %31 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%30 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %32 = polygeist.submapInverse(%0, %31, %10, %11) {map = #map} : (tensor<?x22xf64>, tensor<?x?xf64>, index, index) -> tensor<?x22xf64>
    %33 = polygeist.submap(%19, %7, %10, %11) {map = #map3} : (tensor<?x18xf64>, index, index, index) -> tensor<?x?x?xf64>
    %34 = polygeist.submap(%28, %7, %10, %11) {map = #map4} : (tensor<?x22xf64>, index, index, index) -> tensor<?x?x?xf64>
    %35 = polygeist.submap(%32, %7, %10, %11) {map = #map2} : (tensor<?x22xf64>, index, index, index) -> tensor<?x?x?xf64>
    %36 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%33, %34 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%35 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %39 = arith.mulf %in, %in_0 : f64
      %40 = arith.addf %out, %39 : f64
      linalg.yield %40 : f64
    } -> tensor<?x?x?xf64>
    %37 = polygeist.submapInverse(%32, %36, %7, %10, %11) {map = #map2} : (tensor<?x22xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x22xf64>
    %38 = bufferization.to_memref %37 : memref<?x22xf64>
    memref.copy %38, %arg11 : memref<?x22xf64> to memref<?x22xf64>
    return
  }
}

