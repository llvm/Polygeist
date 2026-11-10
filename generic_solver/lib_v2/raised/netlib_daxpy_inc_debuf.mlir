#map = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_daxpy_inc(%arg0: i32, %arg1: f64, %arg2: memref<?xf64>, %arg3: i32, %arg4: memref<?xf64>, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg5 : i32 to index
    %5 = polygeist.submap(%0, %3, %2) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %6 = polygeist.submap(%1, %4, %2) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%5 : tensor<?xf64>) outs(%6 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %10 = arith.mulf %arg1, %in : f64
      %11 = arith.addf %out, %10 : f64
      linalg.yield %11 : f64
    } -> tensor<?xf64>
    %8 = polygeist.submapInverse(%1, %7, %4, %2) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
    %9 = bufferization.to_memref %8 : memref<?xf64>
    memref.copy %9, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}

