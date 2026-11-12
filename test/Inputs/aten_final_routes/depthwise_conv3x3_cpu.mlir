#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (0, d0, d3 + d1 - 1, d4 + d2 - 1)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (0, d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map7 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
#map8 = affine_map<(d0, d1) -> (-d0 - d1 + 16)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @renamed_kernel(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x8x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x8x16x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x3x3xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?x8x16x16xf32>
    %4 = polygeist.submap(%2, %c8, %c16, %c16) {map = #map} : (tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %5 = polygeist.submap(%3, %c8, %c16, %c16) {map = #map1} : (tensor<?x8x16x16xf32>, index, index, index) -> tensor<?x?x?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%4 : tensor<?x?x?xf32>) outs(%5 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>
    %7 = polygeist.submapInverse(%3, %6, %c8, %c16, %c16) {map = #map1} : (tensor<?x8x16x16xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x8x16x16xf32>
    %8 = polygeist.submap(%0, %c8, %c16, %c16, %c3, %c3) {map = #map3} : (tensor<?x8x16x16xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %9 = polygeist.submap(%1, %c8, %c16, %c16, %c3, %c3) {map = #map4} : (tensor<?x3x3xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %10 = polygeist.submap(%7, %c8, %c16, %c16, %c3, %c3) {map = #map5} : (tensor<?x8x16x16xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %11 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"], library_call = ""} ins(%8, %9 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%10 : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = linalg.index 1 : index
      %15 = linalg.index 2 : index
      %16 = linalg.index 3 : index
      %17 = affine.apply #map7(%14, %16)
      %18 = affine.apply #map8(%14, %16)
      %19 = linalg.index 4 : index
      %20 = affine.apply #map8(%19, %15)
      %21 = arith.cmpi sge, %20, %c0 : index
      %22 = affine.apply #map7(%19, %15)
      %23 = arith.cmpi sge, %22, %c0 : index
      %24 = arith.andi %21, %23 : i1
      %25 = arith.cmpi sge, %17, %c0 : index
      %26 = arith.andi %24, %25 : i1
      %27 = arith.cmpi sge, %18, %c0 : index
      %28 = arith.andi %26, %27 : i1
      %29 = arith.mulf %in, %in_0 : f32
      %30 = arith.addf %out, %29 : f32
      %31 = arith.select %28, %30, %out : f32
      linalg.yield %31 : f32
    } -> tensor<?x?x?x?x?xf32>
    %12 = polygeist.submapInverse(%7, %11, %c8, %c16, %c16, %c3, %c3) {map = #map5} : (tensor<?x8x16x16xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?x8x16x16xf32>
    %13 = bufferization.to_memref %12 : memref<?x8x16x16xf32>
    memref.copy %13, %arg3 : memref<?x8x16x16xf32> to memref<?x8x16x16xf32>
    return
  }
}

