#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1)[s0, s1] -> (s0, s1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0)[s0, s1] -> (s0, s1, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x8x12xf64>, %arg4: memref<?x12xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x12xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?x8x12xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %6:2 = affine.for %arg6 = 0 to %5 iter_args(%arg7 = %2, %arg8 = %0) -> (tensor<?x8x12xf64>, tensor<?xf64>) {
      %9:2 = affine.for %arg9 = 0 to %3 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (tensor<?x8x12xf64>, tensor<?xf64>) {
        %10 = polygeist.submap(%arg11, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
        %11 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%10 : tensor<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        } -> tensor<?xf64>
        %12 = polygeist.submapInverse(%arg11, %11, %4) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
        %13 = polygeist.submap(%arg10, %arg6, %arg9, %4, %4) {map = #map1} : (tensor<?x8x12xf64>, index, index, index, index) -> tensor<?x?xf64>
        %14 = polygeist.submap(%1, %4, %4) {map = #map2} : (tensor<?x12xf64>, index, index) -> tensor<?x?xf64>
        %15 = polygeist.submap(%12, %4, %4) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
        %16 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%13, %14 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%15 : tensor<?x?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %22 = arith.mulf %in, %in_0 : f64
          %23 = arith.addf %out, %22 : f64
          linalg.yield %23 : f64
        } -> tensor<?x?xf64>
        %17 = polygeist.submapInverse(%12, %16, %4, %4) {map = #map3} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %18 = polygeist.submap(%arg10, %arg6, %arg9, %4) {map = #map4} : (tensor<?x8x12xf64>, index, index, index) -> tensor<?xf64>
        %19 = polygeist.submap(%17, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
        %20 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%19 : tensor<?xf64>) outs(%18 : tensor<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          linalg.yield %in : f64
        } -> tensor<?xf64>
        %21 = polygeist.submapInverse(%arg10, %20, %arg6, %arg9, %4) {map = #map4} : (tensor<?x8x12xf64>, tensor<?xf64>, index, index, index) -> tensor<?x8x12xf64>
        affine.yield %21, %17 : tensor<?x8x12xf64>, tensor<?xf64>
      }
      affine.yield %9#0, %9#1 : tensor<?x8x12xf64>, tensor<?xf64>
    }
    %7 = bufferization.to_memref %6#1 : memref<?xf64>
    memref.copy %7, %arg5 : memref<?xf64> to memref<?xf64>
    %8 = bufferization.to_memref %6#0 : memref<?x8x12xf64>
    memref.copy %8, %arg3 : memref<?x8x12xf64> to memref<?x8x12xf64>
    return
  }
}

