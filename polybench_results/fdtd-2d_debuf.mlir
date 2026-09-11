#map = affine_map<(d0) -> (0, d0)>
#map1 = affine_map<(d0)[s0] -> (s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1 + 1, d0)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0 + 1)>
#map7 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x30xf64>, %arg4: memref<?x30xf64>, %arg5: memref<?x30xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.69999999999999996 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?x30xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x30xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?x30xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %7:3 = affine.for %arg7 = 0 to %6 iter_args(%arg8 = %3, %arg9 = %2, %arg10 = %1) -> (tensor<?x30xf64>, tensor<?x30xf64>, tensor<?x30xf64>) {
      %11 = polygeist.submap(%arg9, %5) {map = #map} : (tensor<?x30xf64>, index) -> tensor<?xf64>
      %12 = polygeist.submap(%0, %arg7, %5) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %13 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%12 : tensor<?xf64>) outs(%11 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %14 = polygeist.submapInverse(%arg9, %13, %5) {map = #map} : (tensor<?x30xf64>, tensor<?xf64>, index) -> tensor<?x30xf64>
      %15 = arith.subi %4, %c1 : index
      %16 = polygeist.submap(%14, %5, %15) {map = #map3} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %17 = polygeist.submap(%arg10, %5, %15) {map = #map3} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %18 = polygeist.submap(%arg10, %5, %15) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %19 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%17, %18 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%16 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %42 = arith.subf %in, %in_1 : f64
        %43 = arith.mulf %42, %cst_0 : f64
        %44 = arith.subf %out, %43 : f64
        linalg.yield %44 : f64
      } -> tensor<?x?xf64>
      %20 = polygeist.submapInverse(%14, %19, %5, %15) {map = #map3} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
      %21 = arith.subi %5, %c1 : index
      %22 = arith.subi %5, %c1 : index
      %23 = arith.subi %5, %c1 : index
      %24 = polygeist.submap(%arg8, %23, %4) {map = #map6} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %25 = polygeist.submap(%arg10, %21, %4) {map = #map6} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %26 = polygeist.submap(%arg10, %22, %4) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %27 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%25, %26 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%24 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %42 = arith.subf %in, %in_1 : f64
        %43 = arith.mulf %42, %cst_0 : f64
        %44 = arith.subf %out, %43 : f64
        linalg.yield %44 : f64
      } -> tensor<?x?xf64>
      %28 = polygeist.submapInverse(%arg8, %27, %23, %4) {map = #map6} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
      %29 = affine.apply #map7()[%4]
      %30 = affine.apply #map7()[%5]
      %31 = affine.apply #map7()[%5]
      %32 = affine.apply #map7()[%5]
      %33 = affine.apply #map7()[%5]
      %34 = affine.apply #map7()[%5]
      %35 = polygeist.submap(%28, %30, %29) {map = #map6} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %36 = polygeist.submap(%28, %31, %29) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %37 = polygeist.submap(%20, %32, %29) {map = #map3} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %38 = polygeist.submap(%20, %33, %29) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %39 = polygeist.submap(%arg10, %34, %29) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %40 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%35, %36, %37, %38 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%39 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %42 = arith.subf %in, %in_1 : f64
        %43 = arith.addf %42, %in_2 : f64
        %44 = arith.subf %43, %in_3 : f64
        %45 = arith.mulf %44, %cst : f64
        %46 = arith.subf %out, %45 : f64
        linalg.yield %46 : f64
      } -> tensor<?x?xf64>
      %41 = polygeist.submapInverse(%arg10, %40, %34, %29) {map = #map4} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
      affine.yield %28, %20, %41 : tensor<?x30xf64>, tensor<?x30xf64>, tensor<?x30xf64>
    }
    %8 = bufferization.to_memref %7#2 : memref<?x30xf64>
    memref.copy %8, %arg5 : memref<?x30xf64> to memref<?x30xf64>
    %9 = bufferization.to_memref %7#1 : memref<?x30xf64>
    memref.copy %9, %arg4 : memref<?x30xf64> to memref<?x30xf64>
    %10 = bufferization.to_memref %7#0 : memref<?x30xf64>
    memref.copy %10, %arg3 : memref<?x30xf64> to memref<?x30xf64>
    return
  }
}

