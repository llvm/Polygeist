#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d1 + 1, d0 + 1)>
#map2 = affine_map<(d0, d1) -> (d1 + 1, d0)>
#map3 = affine_map<(d0, d1) -> (d1 + 1, d0 + 2)>
#map4 = affine_map<(d0, d1) -> (d1 + 2, d0 + 1)>
#map5 = affine_map<(d0, d1) -> (d1, d0 + 1)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x30xf64>, %arg3: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x30xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?x30xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4:2 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x30xf64>, tensor<?x30xf64>) {
      %7 = affine.apply #map()[%2]
      %8 = arith.subi %7, %c1 : index
      %9 = affine.apply #map()[%2]
      %10 = arith.subi %9, %c1 : index
      %11 = affine.apply #map()[%2]
      %12 = arith.subi %11, %c1 : index
      %13 = affine.apply #map()[%2]
      %14 = arith.subi %13, %c1 : index
      %15 = affine.apply #map()[%2]
      %16 = arith.subi %15, %c1 : index
      %17 = affine.apply #map()[%2]
      %18 = arith.subi %17, %c1 : index
      %19 = affine.apply #map()[%2]
      %20 = arith.subi %19, %c1 : index
      %21 = polygeist.submap(%arg5, %10, %8) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %22 = polygeist.submap(%arg5, %12, %8) {map = #map2} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %23 = polygeist.submap(%arg5, %14, %8) {map = #map3} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %24 = polygeist.submap(%arg5, %16, %8) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %25 = polygeist.submap(%arg5, %18, %8) {map = #map5} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %26 = polygeist.submap(%arg6, %20, %8) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %27 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%21, %22, %23, %24, %25 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%26 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %51 = arith.addf %in, %in_0 : f64
        %52 = arith.addf %51, %in_1 : f64
        %53 = arith.addf %52, %in_2 : f64
        %54 = arith.addf %53, %in_3 : f64
        %55 = arith.mulf %54, %cst : f64
        linalg.yield %55 : f64
      } -> tensor<?x?xf64>
      %28 = polygeist.submapInverse(%arg6, %27, %20, %8) {map = #map1} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
      %29 = affine.apply #map()[%2]
      %30 = arith.subi %29, %c1 : index
      %31 = affine.apply #map()[%2]
      %32 = arith.subi %31, %c1 : index
      %33 = affine.apply #map()[%2]
      %34 = arith.subi %33, %c1 : index
      %35 = affine.apply #map()[%2]
      %36 = arith.subi %35, %c1 : index
      %37 = affine.apply #map()[%2]
      %38 = arith.subi %37, %c1 : index
      %39 = affine.apply #map()[%2]
      %40 = arith.subi %39, %c1 : index
      %41 = affine.apply #map()[%2]
      %42 = arith.subi %41, %c1 : index
      %43 = polygeist.submap(%arg5, %42, %30) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %44 = polygeist.submap(%28, %32, %30) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %45 = polygeist.submap(%28, %34, %30) {map = #map2} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %46 = polygeist.submap(%28, %36, %30) {map = #map3} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %47 = polygeist.submap(%28, %38, %30) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %48 = polygeist.submap(%28, %40, %30) {map = #map5} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %49 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%44, %45, %46, %47, %48 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%43 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %51 = arith.addf %in, %in_0 : f64
        %52 = arith.addf %51, %in_1 : f64
        %53 = arith.addf %52, %in_2 : f64
        %54 = arith.addf %53, %in_3 : f64
        %55 = arith.mulf %54, %cst : f64
        linalg.yield %55 : f64
      } -> tensor<?x?xf64>
      %50 = polygeist.submapInverse(%arg5, %49, %42, %30) {map = #map1} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
      affine.yield %50, %28 : tensor<?x30xf64>, tensor<?x30xf64>
    }
    %5 = bufferization.to_memref %4#1 : memref<?x30xf64>
    memref.copy %5, %arg3 : memref<?x30xf64> to memref<?x30xf64>
    %6 = bufferization.to_memref %4#0 : memref<?x30xf64>
    memref.copy %6, %arg2 : memref<?x30xf64> to memref<?x30xf64>
    return
  }
}

