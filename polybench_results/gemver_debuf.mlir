#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x40xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg11 : memref<?xf64>
    %1 = bufferization.to_tensor %arg10 : memref<?xf64>
    %2 = bufferization.to_tensor %arg9 : memref<?xf64>
    %3 = bufferization.to_tensor %arg8 : memref<?xf64>
    %4 = bufferization.to_tensor %arg7 : memref<?xf64>
    %5 = bufferization.to_tensor %arg6 : memref<?xf64>
    %6 = bufferization.to_tensor %arg5 : memref<?xf64>
    %7 = bufferization.to_tensor %arg4 : memref<?xf64>
    %8 = bufferization.to_tensor %arg3 : memref<?x40xf64>
    %9 = arith.index_cast %arg0 : i32 to index
    %10 = polygeist.submap(%8, %9, %9) {map = #map} : (tensor<?x40xf64>, index, index) -> tensor<?x?xf64>
    %11 = polygeist.submap(%7, %9, %9) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %12 = polygeist.submap(%6, %9, %9) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %13 = polygeist.submap(%5, %9, %9) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %14 = polygeist.submap(%4, %9, %9) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%11, %12, %13, %14 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%10 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %34 = arith.mulf %in, %in_0 : f64
      %35 = arith.addf %out, %34 : f64
      %36 = arith.mulf %in_1, %in_2 : f64
      %37 = arith.addf %35, %36 : f64
      linalg.yield %37 : f64
    } -> tensor<?x?xf64>
    %16 = polygeist.submapInverse(%8, %15, %9, %9) {map = #map} : (tensor<?x40xf64>, tensor<?x?xf64>, index, index) -> tensor<?x40xf64>
    %17 = bufferization.to_memref %16 : memref<?x40xf64>
    memref.copy %17, %arg3 : memref<?x40xf64> to memref<?x40xf64>
    %18 = polygeist.submap(%16, %9, %9) {map = #map3} : (tensor<?x40xf64>, index, index) -> tensor<?x?xf64>
    %19 = polygeist.submap(%2, %9, %9) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %20 = polygeist.submap(%1, %9, %9) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %21 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%18, %20 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%19 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %34 = arith.mulf %arg2, %in : f64
      %35 = arith.mulf %34, %in_0 : f64
      %36 = arith.addf %out, %35 : f64
      linalg.yield %36 : f64
    } -> tensor<?x?xf64>
    %22 = polygeist.submapInverse(%2, %21, %9, %9) {map = #map1} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %23 = polygeist.submap(%22, %9) {map = #map4} : (tensor<?xf64>, index) -> tensor<?xf64>
    %24 = polygeist.submap(%0, %9) {map = #map4} : (tensor<?xf64>, index) -> tensor<?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map4, #map4], iterator_types = ["parallel"], library_call = ""} ins(%24 : tensor<?xf64>) outs(%23 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %34 = arith.addf %out, %in : f64
      linalg.yield %34 : f64
    } -> tensor<?xf64>
    %26 = polygeist.submapInverse(%22, %25, %9) {map = #map4} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %27 = bufferization.to_memref %26 : memref<?xf64>
    memref.copy %27, %arg9 : memref<?xf64> to memref<?xf64>
    %28 = polygeist.submap(%16, %9, %9) {map = #map} : (tensor<?x40xf64>, index, index) -> tensor<?x?xf64>
    %29 = polygeist.submap(%3, %9, %9) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %30 = polygeist.submap(%26, %9, %9) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %31 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%28, %30 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%29 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %34 = arith.mulf %arg1, %in : f64
      %35 = arith.mulf %34, %in_0 : f64
      %36 = arith.addf %out, %35 : f64
      linalg.yield %36 : f64
    } -> tensor<?x?xf64>
    %32 = polygeist.submapInverse(%3, %31, %9, %9) {map = #map1} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %33 = bufferization.to_memref %32 : memref<?xf64>
    memref.copy %33, %arg8 : memref<?xf64> to memref<?xf64>
    return
  }
}

