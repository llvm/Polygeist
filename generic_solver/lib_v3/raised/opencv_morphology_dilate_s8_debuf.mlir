#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d2 + d0, d3 + d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_morphology_dilate_s8(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?x?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-128_i8 = arith.constant -128 : i8
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x?xi8>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xi8>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = affine.apply #map()[%3]
    %5 = affine.apply #map()[%3]
    %6 = arith.subi %5, %c1 : index
    %7 = arith.subi %4, %c1 : index
    %8 = affine.apply #map()[%2]
    %9 = arith.subi %8, %c1 : index
    %10 = polygeist.submap(%0, %9, %6) {map = #map1} : (tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %11 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%10 : tensor<?x?xi8>) {
    ^bb0(%out: i8):
      linalg.yield %c-128_i8 : i8
    } -> tensor<?x?xi8>
    %12 = polygeist.submapInverse(%0, %11, %9, %6) {map = #map1} : (tensor<?x?xi8>, tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %13 = affine.apply #map()[%2]
    %14 = arith.subi %13, %c1 : index
    %15 = polygeist.submap(%1, %14, %7, %c3, %c3) {map = #map3} : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?x?x?xi8>
    %16 = polygeist.submap(%12, %14, %7, %c3, %c3) {map = #map4} : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?x?x?xi8>
    %17 = linalg.generic {doc = "", indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "reduction", "reduction"], library_call = ""} ins(%15 : tensor<?x?x?x?xi8>) outs(%16 : tensor<?x?x?x?xi8>) {
    ^bb0(%in: i8, %out: i8):
      %20 = arith.extsi %out : i8 to i32
      %21 = arith.extsi %in : i8 to i32
      %22 = arith.cmpi sgt, %20, %21 : i32
      %23 = arith.select %22, %20, %21 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<?x?x?x?xi8>
    %18 = polygeist.submapInverse(%12, %17, %14, %7, %c3, %c3) {map = #map4} : (tensor<?x?xi8>, tensor<?x?x?x?xi8>, index, index, index, index) -> tensor<?x?xi8>
    %19 = bufferization.to_memref %18 : memref<?x?xi8>
    memref.copy %19, %arg3 : memref<?x?xi8> to memref<?x?xi8>
    return
  }
}

