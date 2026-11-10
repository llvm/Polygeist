#map = affine_map<(d0)[s0] -> (d0, s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<(d0, d1)[s0] -> (d1, s0)>
#map4 = affine_map<(d0, d1)[s0] -> (d0, s0)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cholesky_right(%arg0: i32, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = affine.for %arg2 = 0 to %1 iter_args(%arg3 = %0) -> (tensor<?x?xf64>) {
      %extracted = tensor.extract %arg3[%arg2, %arg2] : tensor<?x?xf64>
      %4 = math.sqrt %extracted : f64
      %inserted = tensor.insert %4 into %arg3[%arg2, %arg2] : tensor<?x?xf64>
      %5 = polygeist.submap(%inserted, %arg2, %1) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %6 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%5 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %13 = arith.divf %out, %4 : f64
        %14 = linalg.index 0 : index
        %15 = affine.apply #map2(%arg2)
        %16 = arith.cmpi sge, %14, %15 : index
        %17 = arith.select %16, %13, %out : f64
        linalg.yield %17 : f64
      } -> tensor<?xf64>
      %7 = polygeist.submapInverse(%inserted, %6, %arg2, %1) {map = #map} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %8 = polygeist.submap(%7, %arg2, %1, %1) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %9 = polygeist.submap(%7, %arg2, %1, %1) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %10 = polygeist.submap(%7, %1, %1) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %11 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%8, %9 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%10 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %13 = linalg.index 0 : index
        %14 = arith.mulf %in, %in_0 : f64
        %15 = arith.subf %out, %14 : f64
        %16 = linalg.index 1 : index
        %17 = arith.cmpi sge, %16, %13 : index
        %18 = arith.select %17, %15, %out : f64
        %19 = linalg.index 0 : index
        %20 = affine.apply #map2(%arg2)
        %21 = arith.cmpi sge, %19, %20 : index
        %22 = arith.select %21, %18, %out : f64
        linalg.yield %22 : f64
      } -> tensor<?x?xf64>
      %12 = polygeist.submapInverse(%7, %11, %1, %1) {map = #map5} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      affine.yield %12 : tensor<?x?xf64>
    }
    %3 = bufferization.to_memref %2 : memref<?x?xf64>
    memref.copy %3, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

