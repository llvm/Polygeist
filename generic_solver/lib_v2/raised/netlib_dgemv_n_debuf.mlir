#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dgemv_n(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: f64, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg7 : memref<?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = polygeist.submap(%2, %3) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%4 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %15 = arith.mulf %arg6, %out : f64
      linalg.yield %15 : f64
    } -> tensor<?xf64>
    %6 = polygeist.submapInverse(%2, %5, %3) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = polygeist.submap(%1, %7) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %9 = polygeist.submap(%8, %7, %3) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %10 = polygeist.submap(%0, %7, %3) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %11 = polygeist.submap(%6, %7, %3) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %12 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction", "parallel"], library_call = ""} ins(%9, %10 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%11 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %15 = arith.mulf %arg2, %in : f64
      %16 = arith.mulf %15, %in_0 : f64
      %17 = arith.addf %out, %16 : f64
      linalg.yield %17 : f64
    } -> tensor<?x?xf64>
    %13 = polygeist.submapInverse(%6, %12, %7, %3) {map = #map3} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %14 = bufferization.to_memref %13 : memref<?xf64>
    memref.copy %14, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
}

