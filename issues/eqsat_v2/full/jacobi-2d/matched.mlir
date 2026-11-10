#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map3 = affine_map<(d0, d1) -> (d0 + 1, d1 + 2)>
#map4 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
#map5 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.apply #map()[%2]
    %5 = affine.apply #map()[%2]
    %6 = arith.subi %5, %c1 : index
    %7 = affine.apply #map()[%2]
    %8 = arith.subi %7, %c1 : index
    %9 = arith.subi %4, %c1 : index
    %10 = affine.apply #map()[%2]
    %11 = arith.subi %10, %c1 : index
    %12:2 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x?xf64>, tensor<?x?xf64>) {
      %15 = polygeist.submap(%arg6, %8, %6) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %16 = polygeist.submap(%arg6, %8, %6) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %17 = polygeist.submap(%arg6, %8, %6) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %18 = polygeist.submap(%arg6, %8, %6) {map = #map4} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %19 = polygeist.submap(%arg6, %8, %6) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %20 = polygeist.submap(%arg5, %8, %6) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %21 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%15, %16, %17, %18, %19 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%20 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %31 = arith.addf %in, %in_0 : f64
        %32 = arith.addf %31, %in_1 : f64
        %33 = arith.addf %32, %in_2 : f64
        %34 = arith.addf %33, %in_3 : f64
        %35 = arith.mulf %34, %cst : f64
        linalg.yield %35 : f64
      } -> tensor<?x?xf64>
      %22 = polygeist.submapInverse(%arg5, %21, %8, %6) {map = #map1} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %23 = polygeist.submap(%22, %11, %9) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %24 = polygeist.submap(%22, %11, %9) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %25 = polygeist.submap(%22, %11, %9) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %26 = polygeist.submap(%22, %11, %9) {map = #map4} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %27 = polygeist.submap(%22, %11, %9) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %28 = polygeist.submap(%arg6, %11, %9) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %29 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%23, %24, %25, %26, %27 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%28 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %31 = arith.addf %in, %in_0 : f64
        %32 = arith.addf %31, %in_1 : f64
        %33 = arith.addf %32, %in_2 : f64
        %34 = arith.addf %33, %in_3 : f64
        %35 = arith.mulf %34, %cst : f64
        linalg.yield %35 : f64
      } -> tensor<?x?xf64>
      %30 = polygeist.submapInverse(%arg6, %29, %11, %9) {map = #map1} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      affine.yield %22, %30 : tensor<?x?xf64>, tensor<?x?xf64>
    }
    %13 = bufferization.to_memref %12#0 : memref<?x?xf64>
    memref.copy %13, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %14 = bufferization.to_memref %12#1 : memref<?x?xf64>
    memref.copy %14, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

