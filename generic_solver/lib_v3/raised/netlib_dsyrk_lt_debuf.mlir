#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyrk_lt(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: f64, %arg6: i32, %arg7: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg7 : memref<?x?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = polygeist.submap(%1, %2, %2) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%4 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %13 = linalg.index 0 : index
      %14 = arith.mulf %arg5, %out : f64
      %15 = linalg.index 1 : index
      %16 = arith.cmpi sge, %15, %13 : index
      %17 = arith.select %16, %14, %out : f64
      linalg.yield %17 : f64
    } -> tensor<?x?xf64>
    %6 = polygeist.submapInverse(%1, %5, %2, %2) {map = #map} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %7 = polygeist.submap(%0, %2, %2, %3) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %8 = polygeist.submap(%0, %2, %2, %3) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %9 = polygeist.submap(%6, %2, %2, %3) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%7, %8 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%9 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %13 = linalg.index 0 : index
      %14 = arith.mulf %in, %in_0 : f64
      %15 = arith.mulf %arg2, %14 : f64
      %16 = arith.addf %out, %15 : f64
      %17 = linalg.index 1 : index
      %18 = arith.cmpi sge, %17, %13 : index
      %19 = arith.select %18, %16, %out : f64
      linalg.yield %19 : f64
    } -> tensor<?x?x?xf64>
    %11 = polygeist.submapInverse(%6, %10, %2, %2, %3) {map = #map3} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %12 = bufferization.to_memref %11 : memref<?x?xf64>
    memref.copy %12, %arg7 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

