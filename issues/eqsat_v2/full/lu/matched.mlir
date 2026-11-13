#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0)[s0] -> (d0, s0)>
#map3 = affine_map<(d0)[s0, s1] -> (s0, s1)>
#map4 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0, d1)[s0] -> (s0, d0)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_lu(%arg0: i32, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = affine.for %arg2 = 0 to %1 iter_args(%arg3 = %0) -> (tensor<?x?xf64>) {
      %5 = arith.subi %arg2, %c1 : index
      %6 = affine.for %arg4 = 0 to #map(%arg2) iter_args(%arg5 = %arg3) -> (tensor<?x?xf64>) {
        %12 = polygeist.submap(%arg5, %arg2, %5) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %13 = polygeist.submap(%arg5, %arg4, %5) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %14 = polygeist.submap(%arg5, %arg2, %arg4, %5) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
        %15 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%12, %13 : tensor<?xf64>, tensor<?xf64>) outs(%14 : tensor<?xf64>) {
        ^bb0(%in: f64, %in_1: f64, %out: f64):
          %18 = arith.mulf %in, %in_1 : f64
          %19 = arith.subf %out, %18 : f64
          %20 = linalg.index 0 : index
          %21 = arith.cmpi slt, %20, %arg4 : index
          %22 = arith.select %21, %19, %out : f64
          linalg.yield %22 : f64
        } -> tensor<?xf64>
        %16 = polygeist.submapInverse(%arg5, %15, %arg2, %arg4, %5) {map = #map3} : (tensor<?x?xf64>, tensor<?xf64>, index, index, index) -> tensor<?x?xf64>
        %extracted = tensor.extract %16[%arg4, %arg4] : tensor<?x?xf64>
        %extracted_0 = tensor.extract %16[%arg2, %arg4] : tensor<?x?xf64>
        %17 = arith.divf %extracted_0, %extracted : f64
        %inserted = tensor.insert %17 into %16[%arg2, %arg4] : tensor<?x?xf64>
        affine.yield %inserted : tensor<?x?xf64>
      }
      %7 = polygeist.submap(%6, %arg2, %1, %2) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %8 = polygeist.submap(%6, %1, %2) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %9 = polygeist.submap(%6, %arg2, %1, %2) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %10 = linalg.generic {doc = "", indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%7, %8 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%9 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %12 = arith.mulf %in, %in_0 : f64
        %13 = arith.subf %out, %12 : f64
        %14 = linalg.index 1 : index
        %15 = arith.cmpi slt, %14, %arg2 : index
        %16 = arith.select %15, %13, %out : f64
        %17 = linalg.index 0 : index
        %18 = arith.cmpi sge, %17, %arg2 : index
        %19 = arith.select %18, %16, %out : f64
        linalg.yield %19 : f64
      } -> tensor<?x?xf64>
      %11 = polygeist.submapInverse(%6, %10, %arg2, %1, %2) {map = #map6} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      affine.yield %11 : tensor<?x?xf64>
    }
    %4 = bufferization.to_memref %3 : memref<?x?xf64>
    memref.copy %4, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

