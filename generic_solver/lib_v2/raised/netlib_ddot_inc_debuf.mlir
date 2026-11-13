#map = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_ddot_inc(%arg0: i32, %arg1: memref<?xf64>, %arg2: i32, %arg3: memref<?xf64>, %arg4: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg2 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = tensor.empty() : tensor<f64>
    %inserted = tensor.insert %cst into %5[] : tensor<f64>
    %6 = polygeist.submap(%0, %3, %2) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %7 = polygeist.submap(%1, %4, %2) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %8 = polygeist.submap(%inserted, %2) {map = #map1} : (tensor<f64>, index) -> tensor<?xf64>
    %9 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction"], library_call = ""} ins(%6, %7 : tensor<?xf64>, tensor<?xf64>) outs(%8 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %11 = arith.mulf %in, %in_0 : f64
      %12 = arith.addf %out, %11 : f64
      linalg.yield %12 : f64
    } -> tensor<?xf64>
    %10 = polygeist.submapInverse(%inserted, %9, %2) {map = #map1} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
    %extracted = tensor.extract %10[] : tensor<f64>
    return %extracted : f64
  }
}

