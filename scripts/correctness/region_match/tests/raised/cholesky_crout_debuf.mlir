#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0)[s0] -> (s0, s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map5 = affine_map<(d0, d1)[s0] -> (d0, s0)>
#map6 = affine_map<(d0) -> (d0 + 1)>
#map7 = affine_map<(d0)[s0] -> (d0, s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cholesky_crout(%arg0: i32, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = arith.subi %1, %c1 : index
    %4 = affine.for %arg2 = 0 to %1 iter_args(%arg3 = %0) -> (tensor<?x?xf64>) {
      %6 = polygeist.submap(%arg3, %arg2, %3) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %7 = polygeist.submap(%arg3, %arg2, %3) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %8 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["reduction"], library_call = ""} ins(%6 : tensor<?xf64>) outs(%7 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %19 = arith.mulf %in, %in : f64
        %20 = arith.subf %out, %19 : f64
        %21 = linalg.index 0 : index
        %22 = arith.cmpi slt, %21, %arg2 : index
        %23 = arith.select %22, %20, %out : f64
        linalg.yield %23 : f64
      } -> tensor<?xf64>
      %9 = polygeist.submapInverse(%arg3, %8, %arg2, %3) {map = #map1} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %extracted = tensor.extract %9[%arg2, %arg2] : tensor<?x?xf64>
      %10 = math.sqrt %extracted : f64
      %inserted = tensor.insert %10 into %9[%arg2, %arg2] : tensor<?x?xf64>
      %11 = polygeist.submap(%inserted, %1, %2) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %12 = polygeist.submap(%inserted, %arg2, %1, %2) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %13 = polygeist.submap(%inserted, %arg2, %1, %2) {map = #map5} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %14 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%11, %12 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%13 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %19 = arith.mulf %in, %in_0 : f64
        %20 = arith.subf %out, %19 : f64
        %21 = linalg.index 1 : index
        %22 = arith.cmpi slt, %21, %arg2 : index
        %23 = arith.select %22, %20, %out : f64
        %24 = linalg.index 0 : index
        %25 = affine.apply #map6(%arg2)
        %26 = arith.cmpi sge, %24, %25 : index
        %27 = arith.select %26, %23, %out : f64
        linalg.yield %27 : f64
      } -> tensor<?x?xf64>
      %15 = polygeist.submapInverse(%inserted, %14, %arg2, %1, %2) {map = #map5} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %16 = polygeist.submap(%15, %arg2, %1) {map = #map7} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%16 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %19 = arith.divf %out, %10 : f64
        %20 = linalg.index 0 : index
        %21 = affine.apply #map6(%arg2)
        %22 = arith.cmpi sge, %20, %21 : index
        %23 = arith.select %22, %19, %out : f64
        linalg.yield %23 : f64
      } -> tensor<?xf64>
      %18 = polygeist.submapInverse(%15, %17, %arg2, %1) {map = #map7} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      affine.yield %18 : tensor<?x?xf64>
    }
    %5 = bufferization.to_memref %4 : memref<?x?xf64>
    memref.copy %5, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

