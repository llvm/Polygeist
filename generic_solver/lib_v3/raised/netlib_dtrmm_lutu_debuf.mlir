#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1 - 1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1)[s0] -> (-d0 + s0 - 1, d1)>
#map4 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrmm_lutu(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = affine.apply #map(%c0)[%2]
    %5 = affine.for %arg7 = 0 to %3 iter_args(%arg8 = %0) -> (tensor<?x?xf64>) {
      %alloca = memref.alloca(%2) : memref<?xf64>
      %7 = bufferization.to_tensor %alloca : memref<?xf64>
      %8 = polygeist.submap(%arg8, %arg7, %2, %2) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %9 = polygeist.submap(%7, %2) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %10 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%8 : tensor<?xf64>) outs(%9 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %11 = polygeist.submapInverse(%7, %10, %2) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
      %12 = polygeist.submap(%1, %2, %2, %4) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %13 = polygeist.submap(%arg8, %arg7, %2, %4) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %14 = polygeist.submap(%11, %2, %4) {map = #map5} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%12, %13 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%14 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %21 = linalg.index 0 : index
        %22 = arith.mulf %in, %in_0 : f64
        %23 = arith.addf %out, %22 : f64
        %24 = linalg.index 1 : index
        %25 = affine.apply #map(%21)[%2]
        %26 = arith.cmpi slt, %24, %25 : index
        %27 = arith.select %26, %23, %out : f64
        linalg.yield %27 : f64
      } -> tensor<?x?xf64>
      %16 = polygeist.submapInverse(%11, %15, %2, %4) {map = #map5} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %17 = polygeist.submap(%16, %2) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %18 = polygeist.submap(%arg8, %arg7, %2, %2) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %19 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%17 : tensor<?xf64>) outs(%18 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %21 = arith.mulf %arg2, %in : f64
        linalg.yield %21 : f64
      } -> tensor<?xf64>
      %20 = polygeist.submapInverse(%arg8, %19, %arg7, %2, %2) {map = #map1} : (tensor<?x?xf64>, tensor<?xf64>, index, index, index) -> tensor<?x?xf64>
      affine.yield %20 : tensor<?x?xf64>
    }
    %6 = bufferization.to_memref %5 : memref<?x?xf64>
    memref.copy %6, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

