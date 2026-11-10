#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (s0)>
#map2 = affine_map<(d0) -> (0, d0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.69999999999999996 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg6 : memref<?xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.subi %4, %c1 : index
    %8 = arith.subi %5, %c1 : index
    %9 = affine.apply #map()[%5]
    %10 = affine.apply #map()[%4]
    %11:3 = affine.for %arg7 = 0 to %6 iter_args(%arg8 = %1, %arg9 = %0, %arg10 = %2) -> (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) {
      %15 = polygeist.submap(%3, %arg7, %5) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %16 = polygeist.submap(%arg8, %5) {map = #map2} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%15 : tensor<?xf64>) outs(%16 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %18 = polygeist.submapInverse(%arg8, %17, %5) {map = #map2} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %19 = polygeist.submap(%arg10, %7, %5) {map = #map4} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %20 = polygeist.submap(%arg10, %7, %5) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %21 = polygeist.submap(%18, %7, %5) {map = #map4} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %22 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%19, %20 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%21 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %36 = arith.subf %in, %in_1 : f64
        %37 = arith.mulf %36, %cst_0 : f64
        %38 = arith.subf %out, %37 : f64
        linalg.yield %38 : f64
      } -> tensor<?x?xf64>
      %23 = polygeist.submapInverse(%18, %22, %7, %5) {map = #map4} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %24 = polygeist.submap(%arg10, %4, %8) {map = #map6} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %25 = polygeist.submap(%arg10, %4, %8) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %26 = polygeist.submap(%arg9, %4, %8) {map = #map6} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %27 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%24, %25 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%26 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %36 = arith.subf %in, %in_1 : f64
        %37 = arith.mulf %36, %cst_0 : f64
        %38 = arith.subf %out, %37 : f64
        linalg.yield %38 : f64
      } -> tensor<?x?xf64>
      %28 = polygeist.submapInverse(%arg9, %27, %4, %8) {map = #map6} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %29 = polygeist.submap(%28, %10, %9) {map = #map6} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %30 = polygeist.submap(%28, %10, %9) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %31 = polygeist.submap(%23, %10, %9) {map = #map4} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %32 = polygeist.submap(%23, %10, %9) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %33 = polygeist.submap(%arg10, %10, %9) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %34 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%29, %30, %31, %32 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%33 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %36 = arith.subf %in, %in_1 : f64
        %37 = arith.addf %36, %in_2 : f64
        %38 = arith.subf %37, %in_3 : f64
        %39 = arith.mulf %38, %cst : f64
        %40 = arith.subf %out, %39 : f64
        linalg.yield %40 : f64
      } -> tensor<?x?xf64>
      %35 = polygeist.submapInverse(%arg10, %34, %10, %9) {map = #map5} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      affine.yield %23, %28, %35 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>
    }
    %12 = bufferization.to_memref %11#0 : memref<?x?xf64>
    memref.copy %12, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    %13 = bufferization.to_memref %11#2 : memref<?x?xf64>
    memref.copy %13, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %14 = bufferization.to_memref %11#1 : memref<?x?xf64>
    memref.copy %14, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

