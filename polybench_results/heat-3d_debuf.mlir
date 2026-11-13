#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d2 + 2, d1 + 1, d0 + 1)>
#map2 = affine_map<(d0, d1, d2) -> (d2 + 1, d1 + 1, d0 + 1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1 + 1, d0 + 1)>
#map4 = affine_map<(d0, d1, d2) -> (d2 + 1, d1 + 2, d0 + 1)>
#map5 = affine_map<(d0, d1, d2) -> (d2 + 1, d1, d0 + 1)>
#map6 = affine_map<(d0, d1, d2) -> (d2 + 1, d1 + 1, d0 + 2)>
#map7 = affine_map<(d0, d1, d2) -> (d2 + 1, d1 + 1, d0)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<?x10x10xf64>, %arg3: memref<?x10x10xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.250000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x10x10xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?x10x10xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3:2 = affine.for %arg4 = 1 to 21 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x10x10xf64>, tensor<?x10x10xf64>) {
      %6 = affine.apply #map()[%2]
      %7 = arith.subi %6, %c1 : index
      %8 = affine.apply #map()[%2]
      %9 = arith.subi %8, %c1 : index
      %10 = affine.apply #map()[%2]
      %11 = arith.subi %10, %c1 : index
      %12 = affine.apply #map()[%2]
      %13 = arith.subi %12, %c1 : index
      %14 = affine.apply #map()[%2]
      %15 = arith.subi %14, %c1 : index
      %16 = affine.apply #map()[%2]
      %17 = arith.subi %16, %c1 : index
      %18 = affine.apply #map()[%2]
      %19 = arith.subi %18, %c1 : index
      %20 = affine.apply #map()[%2]
      %21 = arith.subi %20, %c1 : index
      %22 = affine.apply #map()[%2]
      %23 = arith.subi %22, %c1 : index
      %24 = affine.apply #map()[%2]
      %25 = arith.subi %24, %c1 : index
      %26 = affine.apply #map()[%2]
      %27 = arith.subi %26, %c1 : index
      %28 = affine.apply #map()[%2]
      %29 = arith.subi %28, %c1 : index
      %30 = affine.apply #map()[%2]
      %31 = arith.subi %30, %c1 : index
      %32 = affine.apply #map()[%2]
      %33 = arith.subi %32, %c1 : index
      %34 = affine.apply #map()[%2]
      %35 = arith.subi %34, %c1 : index
      %36 = affine.apply #map()[%2]
      %37 = arith.subi %36, %c1 : index
      %38 = affine.apply #map()[%2]
      %39 = arith.subi %38, %c1 : index
      %40 = polygeist.submap(%arg5, %9, %11, %7) {map = #map1} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %41 = polygeist.submap(%arg5, %13, %15, %7) {map = #map2} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %42 = polygeist.submap(%arg5, %17, %19, %7) {map = #map3} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %43 = polygeist.submap(%arg5, %21, %23, %7) {map = #map4} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %44 = polygeist.submap(%arg5, %25, %27, %7) {map = #map5} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %45 = polygeist.submap(%arg5, %29, %31, %7) {map = #map6} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %46 = polygeist.submap(%arg5, %33, %35, %7) {map = #map7} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %47 = polygeist.submap(%arg6, %37, %39, %7) {map = #map2} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %48 = linalg.generic {doc = "", indexing_maps = [#map8, #map8, #map8, #map8, #map8, #map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%40, %41, %42, %43, %44, %45, %46 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%47 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
        %94 = arith.mulf %in_1, %cst : f64
        %95 = arith.subf %in, %94 : f64
        %96 = arith.addf %95, %in_2 : f64
        %97 = arith.mulf %96, %cst_0 : f64
        %98 = arith.subf %in_3, %94 : f64
        %99 = arith.addf %98, %in_4 : f64
        %100 = arith.mulf %99, %cst_0 : f64
        %101 = arith.addf %97, %100 : f64
        %102 = arith.subf %in_5, %94 : f64
        %103 = arith.addf %102, %in_6 : f64
        %104 = arith.mulf %103, %cst_0 : f64
        %105 = arith.addf %101, %104 : f64
        %106 = arith.addf %105, %in_1 : f64
        linalg.yield %106 : f64
      } -> tensor<?x?x?xf64>
      %49 = polygeist.submapInverse(%arg6, %48, %37, %39, %7) {map = #map2} : (tensor<?x10x10xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x10x10xf64>
      %50 = affine.apply #map()[%2]
      %51 = arith.subi %50, %c1 : index
      %52 = affine.apply #map()[%2]
      %53 = arith.subi %52, %c1 : index
      %54 = affine.apply #map()[%2]
      %55 = arith.subi %54, %c1 : index
      %56 = affine.apply #map()[%2]
      %57 = arith.subi %56, %c1 : index
      %58 = affine.apply #map()[%2]
      %59 = arith.subi %58, %c1 : index
      %60 = affine.apply #map()[%2]
      %61 = arith.subi %60, %c1 : index
      %62 = affine.apply #map()[%2]
      %63 = arith.subi %62, %c1 : index
      %64 = affine.apply #map()[%2]
      %65 = arith.subi %64, %c1 : index
      %66 = affine.apply #map()[%2]
      %67 = arith.subi %66, %c1 : index
      %68 = affine.apply #map()[%2]
      %69 = arith.subi %68, %c1 : index
      %70 = affine.apply #map()[%2]
      %71 = arith.subi %70, %c1 : index
      %72 = affine.apply #map()[%2]
      %73 = arith.subi %72, %c1 : index
      %74 = affine.apply #map()[%2]
      %75 = arith.subi %74, %c1 : index
      %76 = affine.apply #map()[%2]
      %77 = arith.subi %76, %c1 : index
      %78 = affine.apply #map()[%2]
      %79 = arith.subi %78, %c1 : index
      %80 = affine.apply #map()[%2]
      %81 = arith.subi %80, %c1 : index
      %82 = affine.apply #map()[%2]
      %83 = arith.subi %82, %c1 : index
      %84 = polygeist.submap(%arg5, %81, %83, %51) {map = #map2} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %85 = polygeist.submap(%49, %53, %55, %51) {map = #map1} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %86 = polygeist.submap(%49, %57, %59, %51) {map = #map2} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %87 = polygeist.submap(%49, %61, %63, %51) {map = #map3} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %88 = polygeist.submap(%49, %65, %67, %51) {map = #map4} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %89 = polygeist.submap(%49, %69, %71, %51) {map = #map5} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %90 = polygeist.submap(%49, %73, %75, %51) {map = #map6} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %91 = polygeist.submap(%49, %77, %79, %51) {map = #map7} : (tensor<?x10x10xf64>, index, index, index) -> tensor<?x?x?xf64>
      %92 = linalg.generic {doc = "", indexing_maps = [#map8, #map8, #map8, #map8, #map8, #map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%85, %86, %87, %88, %89, %90, %91 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%84 : tensor<?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
        %94 = arith.mulf %in_1, %cst : f64
        %95 = arith.subf %in, %94 : f64
        %96 = arith.addf %95, %in_2 : f64
        %97 = arith.mulf %96, %cst_0 : f64
        %98 = arith.subf %in_3, %94 : f64
        %99 = arith.addf %98, %in_4 : f64
        %100 = arith.mulf %99, %cst_0 : f64
        %101 = arith.addf %97, %100 : f64
        %102 = arith.subf %in_5, %94 : f64
        %103 = arith.addf %102, %in_6 : f64
        %104 = arith.mulf %103, %cst_0 : f64
        %105 = arith.addf %101, %104 : f64
        %106 = arith.addf %105, %in_1 : f64
        linalg.yield %106 : f64
      } -> tensor<?x?x?xf64>
      %93 = polygeist.submapInverse(%arg5, %92, %81, %83, %51) {map = #map2} : (tensor<?x10x10xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x10x10xf64>
      affine.yield %93, %49 : tensor<?x10x10xf64>, tensor<?x10x10xf64>
    }
    %4 = bufferization.to_memref %3#1 : memref<?x10x10xf64>
    memref.copy %4, %arg3 : memref<?x10x10xf64> to memref<?x10x10xf64>
    %5 = bufferization.to_memref %3#0 : memref<?x10x10xf64>
    memref.copy %5, %arg2 : memref<?x10x10xf64> to memref<?x10x10xf64>
    return
  }
}

