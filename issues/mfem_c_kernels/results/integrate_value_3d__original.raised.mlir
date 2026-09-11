#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3)[s0] -> (d3 * 25 + d2 + s0 * 125 + d0 * 5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d0)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
#map9 = affine_map<(d0, d1, d2, d3)[s0] -> (d2 + s0 * 64 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_value_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<5x5x4xf64>
    affine.for %arg3 = 0 to 2 {
      %0 = polygeist.submap(%alloca_0, %c5, %c4, %c5) {map = #map} : (memref<5x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %1 = polygeist.submap(%arg0, %arg3, %c5, %c4, %c5, %c5) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?xf64>
      %2 = polygeist.submap(%arg1, %c5, %c4, %c5, %c5) {map = #map3} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %3 = polygeist.submap(%alloca_0, %c5, %c4, %c5, %c5) {map = #map4} : (memref<5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%3 : memref<?x?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %11 = arith.mulf %in, %in_1 : f64
        %12 = arith.addf %out, %11 : f64
        linalg.yield %12 : f64
      }
      %4 = polygeist.submap(%alloca, %c4, %c4, %c5) {map = #map} : (memref<5x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %5 = polygeist.submap(%alloca_0, %c4, %c4, %c5, %c5) {map = #map6} : (memref<5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %6 = polygeist.submap(%arg1, %c4, %c4, %c5, %c5) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %7 = polygeist.submap(%alloca, %c4, %c4, %c5, %c5) {map = #map4} : (memref<5x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%7 : memref<?x?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %11 = arith.mulf %in, %in_1 : f64
        %12 = arith.addf %out, %11 : f64
        linalg.yield %12 : f64
      }
      %8 = polygeist.submap(%alloca, %c4, %c4, %c4, %c5) {map = #map8} : (memref<5x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %9 = polygeist.submap(%arg1, %c4, %c4, %c4, %c5) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %10 = polygeist.submap(%arg2, %arg3, %c4, %c4, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%8, %9 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%10 : memref<?x?x?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %11 = arith.mulf %in, %in_1 : f64
        %12 = arith.addf %out, %11 : f64
        linalg.yield %12 : f64
      }
    }
    return
  }
}
