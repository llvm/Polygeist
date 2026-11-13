#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0)[s0] -> (d0, s0)>
#map6 = affine_map<(d0)[s0, s1] -> (s0, s1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x28xf64>, %arg4: memref<?x28xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x28xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?x28xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = polygeist.submap(%0, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%5 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?xf64>
    %7 = polygeist.submapInverse(%0, %6, %4) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %8 = polygeist.submap(%2, %3, %4) {map = #map1} : (tensor<?x28xf64>, index, index) -> tensor<?x?xf64>
    %9 = polygeist.submap(%7, %3, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%8 : tensor<?x?xf64>) outs(%9 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %24 = arith.addf %out, %in : f64
      linalg.yield %24 : f64
    } -> tensor<?x?xf64>
    %11 = polygeist.submapInverse(%7, %10, %3, %4) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %12 = polygeist.submap(%11, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%12 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %24 = arith.divf %out, %arg2 : f64
      linalg.yield %24 : f64
    } -> tensor<?xf64>
    %14 = polygeist.submapInverse(%11, %13, %4) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %15 = bufferization.to_memref %14 : memref<?xf64>
    memref.copy %15, %arg5 : memref<?xf64> to memref<?xf64>
    %16 = polygeist.submap(%2, %4, %3) {map = #map3} : (tensor<?x28xf64>, index, index) -> tensor<?x?xf64>
    %17 = polygeist.submap(%14, %4, %3) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%17 : tensor<?x?xf64>) outs(%16 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %24 = arith.subf %out, %in : f64
      linalg.yield %24 : f64
    } -> tensor<?x?xf64>
    %19 = polygeist.submapInverse(%2, %18, %4, %3) {map = #map3} : (tensor<?x28xf64>, tensor<?x?xf64>, index, index) -> tensor<?x28xf64>
    %20 = bufferization.to_memref %19 : memref<?x28xf64>
    memref.copy %20, %arg3 : memref<?x28xf64> to memref<?x28xf64>
    %21 = arith.subf %arg2, %cst : f64
    %22 = affine.for %arg6 = 0 to %4 iter_args(%arg7 = %1) -> (tensor<?x28xf64>) {
      %24 = affine.for %arg8 = #map(%arg6) to %4 iter_args(%arg9 = %arg7) -> (tensor<?x28xf64>) {
        %inserted = tensor.insert %cst_0 into %arg9[%arg6, %arg8] : tensor<?x28xf64>
        %25 = polygeist.submap(%19, %arg6, %3) {map = #map5} : (tensor<?x28xf64>, index, index) -> tensor<?xf64>
        %26 = polygeist.submap(%19, %arg8, %3) {map = #map5} : (tensor<?x28xf64>, index, index) -> tensor<?xf64>
        %27 = polygeist.submap(%inserted, %arg6, %arg8, %3) {map = #map6} : (tensor<?x28xf64>, index, index, index) -> tensor<?xf64>
        %28 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%25, %26 : tensor<?xf64>, tensor<?xf64>) outs(%27 : tensor<?xf64>) {
        ^bb0(%in: f64, %in_3: f64, %out: f64):
          %31 = arith.mulf %in, %in_3 : f64
          %32 = arith.addf %out, %31 : f64
          linalg.yield %32 : f64
        } -> tensor<?xf64>
        %29 = polygeist.submapInverse(%inserted, %28, %arg6, %arg8, %3) {map = #map6} : (tensor<?x28xf64>, tensor<?xf64>, index, index, index) -> tensor<?x28xf64>
        %extracted = tensor.extract %29[%arg6, %arg8] : tensor<?x28xf64>
        %30 = arith.divf %extracted, %21 : f64
        %inserted_1 = tensor.insert %30 into %29[%arg6, %arg8] : tensor<?x28xf64>
        %inserted_2 = tensor.insert %30 into %inserted_1[%arg8, %arg6] : tensor<?x28xf64>
        affine.yield %inserted_2 : tensor<?x28xf64>
      }
      affine.yield %24 : tensor<?x28xf64>
    }
    %23 = bufferization.to_memref %22 : memref<?x28xf64>
    memref.copy %23, %arg4 : memref<?x28xf64> to memref<?x28xf64>
    return
  }
}

