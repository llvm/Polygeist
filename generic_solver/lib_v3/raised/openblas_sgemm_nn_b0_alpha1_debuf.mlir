#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @openblas_sgemm_nn_b0_alpha1(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?xf32>, %arg5: i32, %arg6: memref<?x?xf32>, %arg7: i32, %arg8: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg8 : memref<?x?xf32>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf32>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = polygeist.submap(%0, %3, %4) {map = #map} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%6 : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    %8 = polygeist.submapInverse(%0, %7, %3, %4) {map = #map} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %9 = polygeist.submap(%2, %3, %4, %5) {map = #map2} : (tensor<?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %10 = polygeist.submap(%1, %3, %4, %5) {map = #map3} : (tensor<?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %11 = polygeist.submap(%8, %3, %4, %5) {map = #map4} : (tensor<?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %12 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%9, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%11 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %15 = arith.mulf %in, %in_0 : f32
      %16 = arith.addf %out, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<?x?x?xf32>
    %13 = polygeist.submapInverse(%8, %12, %3, %4, %5) {map = #map4} : (tensor<?x?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x?xf32>
    %14 = bufferization.to_memref %13 : memref<?x?xf32>
    memref.copy %14, %arg8 : memref<?x?xf32> to memref<?x?xf32>
    return
  }
}

