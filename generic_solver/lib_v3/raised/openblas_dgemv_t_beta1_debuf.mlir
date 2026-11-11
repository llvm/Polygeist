#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @openblas_dgemv_t_beta1(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg6 : memref<?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = polygeist.submap(%0, %3, %4) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %6 = polygeist.submap(%1, %3, %4) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %7 = polygeist.submap(%2, %3, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%5, %6 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%7 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %11 = arith.mulf %in, %in_0 : f64
      %12 = arith.mulf %arg2, %11 : f64
      %13 = arith.addf %out, %12 : f64
      linalg.yield %13 : f64
    } -> tensor<?x?xf64>
    %9 = polygeist.submapInverse(%2, %8, %3, %4) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %10 = bufferization.to_memref %9 : memref<?xf64>
    memref.copy %10, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}

