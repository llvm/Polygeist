#map = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, -d0 - s0 + s1 * 2 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 - s0 + s1 * 2 - 1)>
#map2 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrsv_ltu(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = affine.for %arg4 = 0 to %2 iter_args(%arg5 = %0) -> (tensor<?xf64>) {
      %5 = polygeist.submap(%1, %arg4, %2, %2) {map = #map} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %6 = polygeist.submap(%arg5, %arg4, %2, %2) {map = #map1} : (tensor<?xf64>, index, index, index) -> tensor<?xf64>
      %7 = polygeist.submap(%arg5, %arg4, %2, %2) {map = #map2} : (tensor<?xf64>, index, index, index) -> tensor<?xf64>
      %8 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["reduction"], library_call = ""} ins(%5, %6 : tensor<?xf64>, tensor<?xf64>) outs(%7 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %10 = arith.mulf %in, %in_0 : f64
        %11 = arith.subf %out, %10 : f64
        %12 = linalg.index 0 : index
        %13 = affine.apply #map4(%arg4)[%2]
        %14 = arith.cmpi sge, %12, %13 : index
        %15 = arith.select %14, %11, %out : f64
        linalg.yield %15 : f64
      } -> tensor<?xf64>
      %9 = polygeist.submapInverse(%arg5, %8, %arg4, %2, %2) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index, index, index) -> tensor<?xf64>
      affine.yield %9 : tensor<?xf64>
    }
    %4 = bufferization.to_memref %3 : memref<?xf64>
    memref.copy %4, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}

