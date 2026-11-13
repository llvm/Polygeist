#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<(d0) -> (d0 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.apply #map()[%2]
    %5 = affine.apply #map()[%2]
    %6 = arith.subi %5, %c1 : index
    %7 = arith.subi %4, %c1 : index
    %8:2 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?xf64>, tensor<?xf64>) {
      %11 = polygeist.submap(%arg6, %6) {map = #map1} : (tensor<?xf64>, index) -> tensor<?xf64>
      %12 = polygeist.submap(%arg6, %6) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %13 = polygeist.submap(%arg6, %6) {map = #map3} : (tensor<?xf64>, index) -> tensor<?xf64>
      %14 = polygeist.submap(%arg5, %6) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%11, %12, %13 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%14 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
        %23 = arith.addf %in, %in_0 : f64
        %24 = arith.addf %23, %in_1 : f64
        %25 = arith.mulf %24, %cst : f64
        linalg.yield %25 : f64
      } -> tensor<?xf64>
      %16 = polygeist.submapInverse(%arg5, %15, %6) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
      %17 = polygeist.submap(%16, %7) {map = #map1} : (tensor<?xf64>, index) -> tensor<?xf64>
      %18 = polygeist.submap(%16, %7) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %19 = polygeist.submap(%16, %7) {map = #map3} : (tensor<?xf64>, index) -> tensor<?xf64>
      %20 = polygeist.submap(%arg6, %7) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %21 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%17, %18, %19 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%20 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
        %23 = arith.addf %in, %in_0 : f64
        %24 = arith.addf %23, %in_1 : f64
        %25 = arith.mulf %24, %cst : f64
        linalg.yield %25 : f64
      } -> tensor<?xf64>
      %22 = polygeist.submapInverse(%arg6, %21, %7) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
      affine.yield %16, %22 : tensor<?xf64>, tensor<?xf64>
    }
    %9 = bufferization.to_memref %8#1 : memref<?xf64>
    memref.copy %9, %arg2 : memref<?xf64> to memref<?xf64>
    %10 = bufferization.to_memref %8#0 : memref<?xf64>
    memref.copy %10, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}

