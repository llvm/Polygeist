#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d0 + 2, d1 + 1, d2 + 1)>
#map2 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 1, d2 + 1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1 + 1, d2 + 1)>
#map4 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 2, d2 + 1)>
#map5 = affine_map<(d0, d1, d2) -> (d0 + 1, d1, d2 + 1)>
#map6 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 1, d2 + 2)>
#map7 = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<?x?x?xf64>, %arg3: memref<?x?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.250000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x?x?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?x?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.apply #map()[%2]
    %4 = affine.apply #map()[%2]
    %5 = arith.subi %4, %c1 : index
    %6 = affine.apply #map()[%2]
    %7 = arith.subi %6, %c1 : index
    %8 = affine.apply #map()[%2]
    %9 = arith.subi %8, %c1 : index
    %10 = arith.subi %3, %c1 : index
    %11 = affine.apply #map()[%2]
    %12:2 = affine.for %arg4 = 1 to 501 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
      %15 = polygeist.submap(%arg6, %9, %7, %5) {map = #map1} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %16 = polygeist.submap(%arg6, %9, %7, %5) {map = #map2} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %17 = polygeist.submap(%arg6, %9, %7, %5) {map = #map3} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %18 = polygeist.submap(%arg6, %9, %7, %5) {map = #map4} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %19 = polygeist.submap(%arg6, %9, %7, %5) {map = #map5} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %20 = polygeist.submap(%arg6, %9, %7, %5) {map = #map6} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %21 = polygeist.submap(%arg6, %9, %7, %5) {map = #map7} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %22 = polygeist.submap(%arg5, %9, %7, %5) {map = #map2} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %23 = linalg.generic {doc = "", indexing_maps = [#map8, #map8, #map8, #map8, #map8, #map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%15, %16, %17, %18, %19, %20, %21 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%22 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
        %38 = arith.mulf %in_1, %cst : f64
        %39 = arith.subf %in, %38 : f64
        %40 = arith.addf %39, %in_2 : f64
        %41 = arith.mulf %40, %cst_0 : f64
        %42 = arith.subf %in_3, %38 : f64
        %43 = arith.addf %42, %in_4 : f64
        %44 = arith.mulf %43, %cst_0 : f64
        %45 = arith.addf %41, %44 : f64
        %46 = arith.subf %in_5, %38 : f64
        %47 = arith.addf %46, %in_6 : f64
        %48 = arith.mulf %47, %cst_0 : f64
        %49 = arith.addf %45, %48 : f64
        %50 = arith.addf %49, %in_1 : f64
        linalg.yield %50 : f64
      } -> tensor<?x?x?xf64>
      %24 = polygeist.submapInverse(%arg5, %23, %9, %7, %5) {map = #map2} : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %25 = arith.subi %11, %c1 : index
      %26 = affine.apply #map()[%2]
      %27 = arith.subi %26, %c1 : index
      %28 = polygeist.submap(%24, %27, %25, %10) {map = #map1} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %29 = polygeist.submap(%24, %27, %25, %10) {map = #map2} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %30 = polygeist.submap(%24, %27, %25, %10) {map = #map3} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %31 = polygeist.submap(%24, %27, %25, %10) {map = #map4} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %32 = polygeist.submap(%24, %27, %25, %10) {map = #map5} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %33 = polygeist.submap(%24, %27, %25, %10) {map = #map6} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %34 = polygeist.submap(%24, %27, %25, %10) {map = #map7} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %35 = polygeist.submap(%arg6, %27, %25, %10) {map = #map2} : (tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      %36 = linalg.generic {doc = "", indexing_maps = [#map8, #map8, #map8, #map8, #map8, #map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%28, %29, %30, %31, %32, %33, %34 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%35 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
        %38 = arith.mulf %in_1, %cst : f64
        %39 = arith.subf %in, %38 : f64
        %40 = arith.addf %39, %in_2 : f64
        %41 = arith.mulf %40, %cst_0 : f64
        %42 = arith.subf %in_3, %38 : f64
        %43 = arith.addf %42, %in_4 : f64
        %44 = arith.mulf %43, %cst_0 : f64
        %45 = arith.addf %41, %44 : f64
        %46 = arith.subf %in_5, %38 : f64
        %47 = arith.addf %46, %in_6 : f64
        %48 = arith.mulf %47, %cst_0 : f64
        %49 = arith.addf %45, %48 : f64
        %50 = arith.addf %49, %in_1 : f64
        linalg.yield %50 : f64
      } -> tensor<?x?x?xf64>
      %37 = polygeist.submapInverse(%arg6, %36, %27, %25, %10) {map = #map2} : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
      affine.yield %24, %37 : tensor<?x?x?xf64>, tensor<?x?x?xf64>
    }
    %13 = bufferization.to_memref %12#0 : memref<?x?x?xf64>
    memref.copy %13, %arg3 : memref<?x?x?xf64> to memref<?x?x?xf64>
    %14 = bufferization.to_memref %12#1 : memref<?x?x?xf64>
    memref.copy %14, %arg2 : memref<?x?x?xf64> to memref<?x?x?xf64>
    return
  }
}

