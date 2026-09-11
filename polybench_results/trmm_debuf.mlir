#map = affine_map<(d0, d1)[s0] -> (d0, s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0)[s0] -> (s0, d0)>
#map5 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x20xf64>, %arg4: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg4 : memref<?x30xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?x20xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.for %arg5 = 0 to %3 iter_args(%arg6 = %0) -> (tensor<?x30xf64>) {
      %6 = polygeist.submap(%1, %arg5, %3, %2) {map = #map} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %7 = polygeist.submap(%arg6, %3, %2) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
      %8 = polygeist.submap(%arg6, %arg5, %3, %2) {map = #map2} : (tensor<?x30xf64>, index, index, index) -> tensor<?x?xf64>
      %9 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%6, %7 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%8 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %14 = arith.mulf %in, %in_0 : f64
        %15 = arith.addf %out, %14 : f64
        %16 = linalg.index 1 : index
        %17 = affine.apply #map3(%arg5)
        %18 = arith.cmpi sge, %16, %17 : index
        %19 = arith.select %18, %15, %out : f64
        linalg.yield %19 : f64
      } -> tensor<?x?xf64>
      %10 = polygeist.submapInverse(%arg6, %9, %arg5, %3, %2) {map = #map2} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x30xf64>
      %11 = polygeist.submap(%10, %arg5, %2) {map = #map4} : (tensor<?x30xf64>, index, index) -> tensor<?xf64>
      %12 = linalg.generic {doc = "", indexing_maps = [#map5], iterator_types = ["parallel"], library_call = ""} outs(%11 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %14 = arith.mulf %arg2, %out : f64
        linalg.yield %14 : f64
      } -> tensor<?xf64>
      %13 = polygeist.submapInverse(%10, %12, %arg5, %2) {map = #map4} : (tensor<?x30xf64>, tensor<?xf64>, index, index) -> tensor<?x30xf64>
      affine.yield %13 : tensor<?x30xf64>
    }
    %5 = bufferization.to_memref %4 : memref<?x30xf64>
    memref.copy %5, %arg4 : memref<?x30xf64> to memref<?x30xf64>
    return
  }
}

