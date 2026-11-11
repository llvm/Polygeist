#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @openblas_daxpby(%arg0: i32, %arg1: f64, %arg2: memref<?xf64>, %arg3: f64, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %4 = polygeist.submap(%1, %2) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%3 : tensor<?xf64>) outs(%4 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %8 = arith.mulf %arg1, %in : f64
      %9 = arith.mulf %arg3, %out : f64
      %10 = arith.addf %8, %9 : f64
      linalg.yield %10 : f64
    } -> tensor<?xf64>
    %6 = polygeist.submapInverse(%1, %5, %2) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %7 = bufferization.to_memref %6 : memref<?xf64>
    memref.copy %7, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}

