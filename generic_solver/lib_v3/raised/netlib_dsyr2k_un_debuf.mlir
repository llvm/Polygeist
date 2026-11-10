#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyr2k_un(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.subi %3, %c1 : index
    %6 = affine.apply #map(%5)
    %7 = polygeist.submap(%2, %3, %6) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%7 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %20 = linalg.index 0 : index
      %21 = arith.mulf %arg7, %out : f64
      %22 = linalg.index 1 : index
      %23 = affine.apply #map(%20)
      %24 = arith.cmpi slt, %22, %23 : index
      %25 = arith.select %24, %21, %out : f64
      linalg.yield %25 : f64
    } -> tensor<?x?xf64>
    %9 = polygeist.submapInverse(%2, %8, %3, %6) {map = #map1} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %10 = arith.subi %3, %c1 : index
    %11 = affine.apply #map(%10)
    %12 = polygeist.submap(%1, %3, %4, %11) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %13 = polygeist.submap(%0, %3, %4, %11) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %14 = polygeist.submap(%0, %3, %4, %11) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %15 = polygeist.submap(%1, %3, %4, %11) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %16 = polygeist.submap(%9, %3, %4, %11) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %17 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%12, %13, %14, %15 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%16 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %20 = linalg.index 0 : index
      %21 = arith.mulf %arg2, %in : f64
      %22 = arith.mulf %arg2, %in_0 : f64
      %23 = arith.mulf %in_1, %21 : f64
      %24 = arith.addf %out, %23 : f64
      %25 = arith.mulf %in_2, %22 : f64
      %26 = arith.addf %24, %25 : f64
      %27 = linalg.index 2 : index
      %28 = affine.apply #map(%20)
      %29 = arith.cmpi slt, %27, %28 : index
      %30 = arith.select %29, %26, %out : f64
      linalg.yield %30 : f64
    } -> tensor<?x?x?xf64>
    %18 = polygeist.submapInverse(%9, %17, %3, %4, %11) {map = #map4} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %19 = bufferization.to_memref %18 : memref<?x?xf64>
    memref.copy %19, %arg9 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

