#map = affine_map<(d0) -> (d0, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map8 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ru(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.subi %3, %c1 : index
    %6 = polygeist.submap(%0, %3) {map = #map} : (tensor<?x?xf64>, index) -> tensor<?xf64>
    %7 = polygeist.submap(%6, %3, %4) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %8 = polygeist.submap(%1, %3, %4) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %9 = polygeist.submap(%2, %3, %4) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%7, %8 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%9 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %23 = arith.mulf %arg2, %in : f64
      %24 = arith.mulf %arg7, %out : f64
      %25 = arith.mulf %23, %in_0 : f64
      %26 = arith.addf %24, %25 : f64
      linalg.yield %26 : f64
    } -> tensor<?x?xf64>
    %11 = polygeist.submapInverse(%2, %10, %3, %4) {map = #map2} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %12 = polygeist.submap(%0, %3, %5, %4) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %13 = polygeist.submap(%1, %3, %5, %4) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %14 = polygeist.submap(%11, %3, %5, %4) {map = #map5} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%12, %13 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%14 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %23 = linalg.index 0 : index
      %24 = arith.mulf %arg2, %in : f64
      %25 = arith.mulf %24, %in_0 : f64
      %26 = arith.addf %out, %25 : f64
      %27 = linalg.index 1 : index
      %28 = arith.cmpi slt, %27, %23 : index
      %29 = arith.select %28, %26, %out : f64
      linalg.yield %29 : f64
    } -> tensor<?x?x?xf64>
    %16 = polygeist.submapInverse(%11, %15, %3, %5, %4) {map = #map5} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %17 = polygeist.submap(%0, %3, %3, %4) {map = #map7} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %18 = polygeist.submap(%1, %3, %3, %4) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %19 = polygeist.submap(%16, %3, %3, %4) {map = #map5} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %20 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%17, %18 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%19 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %23 = linalg.index 0 : index
      %24 = arith.mulf %arg2, %in : f64
      %25 = arith.mulf %24, %in_0 : f64
      %26 = arith.addf %out, %25 : f64
      %27 = linalg.index 1 : index
      %28 = affine.apply #map8(%23)
      %29 = arith.cmpi sge, %27, %28 : index
      %30 = arith.select %29, %26, %out : f64
      linalg.yield %30 : f64
    } -> tensor<?x?x?xf64>
    %21 = polygeist.submapInverse(%16, %20, %3, %3, %4) {map = #map5} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %22 = bufferization.to_memref %21 : memref<?x?xf64>
    memref.copy %22, %arg9 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

