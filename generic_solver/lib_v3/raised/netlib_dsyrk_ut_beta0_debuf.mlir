#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyrk_ut_beta0(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.subi %2, %c1 : index
    %5 = arith.subi %2, %c1 : index
    %6 = affine.apply #map(%5)
    %7 = affine.apply #map(%4)
    %8 = polygeist.submap(%0, %2, %6) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %9 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%8 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %17 = linalg.index 0 : index
      %18 = arith.mulf %arg2, %cst : f64
      %19 = linalg.index 1 : index
      %20 = affine.apply #map(%17)
      %21 = arith.cmpi slt, %19, %20 : index
      %22 = arith.select %21, %18, %out : f64
      linalg.yield %22 : f64
    } -> tensor<?x?xf64>
    %10 = polygeist.submapInverse(%0, %9, %2, %6) {map = #map1} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %11 = polygeist.submap(%1, %2, %7, %3) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %12 = polygeist.submap(%1, %2, %7, %3) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %13 = polygeist.submap(%10, %2, %7, %3) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %14 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%11, %12 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%13 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = linalg.index 0 : index
      %18 = arith.mulf %in, %in_0 : f64
      %19 = arith.mulf %arg2, %18 : f64
      %20 = arith.addf %out, %19 : f64
      %21 = linalg.index 1 : index
      %22 = affine.apply #map(%17)
      %23 = arith.cmpi slt, %21, %22 : index
      %24 = arith.select %23, %20, %out : f64
      linalg.yield %24 : f64
    } -> tensor<?x?x?xf64>
    %15 = polygeist.submapInverse(%10, %14, %2, %7, %3) {map = #map4} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %16 = bufferization.to_memref %15 : memref<?x?xf64>
    memref.copy %16, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

