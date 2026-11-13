#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1 * 4 + d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0)[s0, s1] -> (d0 + s0 * 16 + s1 * 4)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map7 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 25 + d0 * 5)>
#map8 = affine_map<(d0, d1) -> (d1 * 5 + d0)>
#map9 = affine_map<(d0)[s0] -> (s0, d0)>
#map10 = affine_map<(d0, d1)[s0] -> (d0 * 5 + s0)>
#map11 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 16 + d0 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_mass_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<5xf64>
    %alloca_1 = memref.alloca() : memref<5x5xf64>
    affine.for %arg5 = 0 to 2 {
      %0 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : memref<?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg6 = 0 to 4 {
        %3 = polygeist.submap(%alloca_0, %c5) {map = #map1} : (memref<5xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %4 = polygeist.submap(%arg0, %c4, %c5) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        %5 = polygeist.submap(%alloca_0, %c4, %c5) {map = #map3} : (memref<5xf64>, index, index) -> memref<?x?xf64>
        %6 = polygeist.submap(%arg3, %arg5, %arg6, %c4) {map = #map4} : (memref<?xf64>, index, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%6, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction", "parallel"]} ins(%7, %4 : memref<?x?xf64>, memref<?x?xf64>) outs(%5 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
          %12 = arith.mulf %in_2, %in : f64
          %13 = arith.addf %out, %12 : f64
          linalg.yield %13 : f64
        }
        %8 = polygeist.submap(%alloca_0, %c5, %c5) {map = #map3} : (memref<5xf64>, index, index) -> memref<?x?xf64>
        %9 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
        %10 = polygeist.submap(%arg0, %arg6, %c5) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
        %11 = polygeist.submap(%10, %c5, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %8 : memref<?x?xf64>, memref<?x?xf64>) outs(%9 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
          %12 = arith.mulf %in, %in_2 : f64
          %13 = arith.addf %out, %12 : f64
          linalg.yield %13 : f64
        }
      }
      %1 = polygeist.submap(%arg2, %arg5, %c5, %c5) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %2 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<?x?xf64>) outs(%2 : memref<?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %3 = arith.mulf %out, %in : f64
        linalg.yield %3 : f64
      }
      affine.for %arg6 = 0 to 5 {
        %3 = polygeist.submap(%alloca, %c4) {map = #map1} : (memref<4xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %4 = polygeist.submap(%arg1, %c5, %c4) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        %5 = polygeist.submap(%alloca, %c5, %c4) {map = #map3} : (memref<4xf64>, index, index) -> memref<?x?xf64>
        %6 = polygeist.submap(%alloca_1, %arg6, %c5) {map = #map9} : (memref<5x5xf64>, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%6, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction", "parallel"]} ins(%7, %4 : memref<?x?xf64>, memref<?x?xf64>) outs(%5 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
          %11 = arith.mulf %in_2, %in : f64
          %12 = arith.addf %out, %11 : f64
          linalg.yield %12 : f64
        }
        %8 = polygeist.submap(%arg1, %arg6, %c4, %c4) {map = #map10} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        %9 = polygeist.submap(%alloca, %c4, %c4) {map = #map3} : (memref<4xf64>, index, index) -> memref<?x?xf64>
        %10 = polygeist.submap(%arg4, %arg5, %c4, %c4) {map = #map11} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : memref<?x?xf64>, memref<?x?xf64>) outs(%10 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
          %11 = arith.mulf %in, %in_2 : f64
          %12 = arith.addf %out, %11 : f64
          linalg.yield %12 : f64
        }
      }
    }
    return
  }
}
