#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1 * 4 + d0)>
#map4 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 64 + s1 * 16 + s2 * 4)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map8 = affine_map<(d0, d1, d2) -> (d0)>
#map9 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map10 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 125 + d0 * 25 + d1 * 5)>
#map11 = affine_map<(d0, d1) -> (d1 * 5 + d0)>
#map12 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map13 = affine_map<(d0, d1, d2)[s0] -> (d0 * 5 + s0)>
#map14 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 64 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_mass_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<4x4xf64>
    %alloca_1 = memref.alloca() : memref<5xf64>
    %alloca_2 = memref.alloca() : memref<5x5xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5xf64>
    affine.for %arg5 = 0 to 2 {
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_3 : memref<5x5x5xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg6 = 0 to 4 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_2 : memref<5x5xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg7 = 0 to 4 {
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca_1 : memref<5xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%arg0, %c4, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %3 = polygeist.submap(%arg3, %arg5, %arg6, %arg7, %c4) {map = #map4} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map5, #map1, #map6], iterator_types = ["reduction", "parallel"]} ins(%3, %2 : memref<?xf64>, memref<?x?xf64>) outs(%alloca_1 : memref<5xf64>) {
          ^bb0(%in: f64, %in_4: f64, %out: f64):
            %5 = arith.mulf %in_4, %in : f64
            %6 = arith.addf %out, %5 : f64
            linalg.yield %6 : f64
          }
          %4 = polygeist.submap(%arg0, %arg7, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map5, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %alloca_1 : memref<?xf64>, memref<5xf64>) outs(%alloca_2 : memref<5x5xf64>) {
          ^bb0(%in: f64, %in_4: f64, %out: f64):
            %5 = arith.mulf %in, %in_4 : f64
            %6 = arith.addf %out, %5 : f64
            linalg.yield %6 : f64
          }
        }
        %1 = polygeist.submap(%arg0, %arg6, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map8, #map9, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %alloca_2 : memref<?xf64>, memref<5x5xf64>) outs(%alloca_3 : memref<5x5x5xf64>) {
        ^bb0(%in: f64, %in_4: f64, %out: f64):
          %2 = arith.mulf %in, %in_4 : f64
          %3 = arith.addf %out, %2 : f64
          linalg.yield %3 : f64
        }
      }
      %0 = polygeist.submap(%arg2, %arg5, %c5, %c5, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : memref<?x?x?xf64>) outs(%alloca_3 : memref<5x5x5xf64>) {
      ^bb0(%in: f64, %out: f64):
        %1 = arith.mulf %out, %in : f64
        linalg.yield %1 : f64
      }
      affine.for %arg6 = 0 to 5 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_0 : memref<4x4xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg7 = 0 to 5 {
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca : memref<4xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %3 = polygeist.submap(%arg1, %c5, %c4) {map = #map11} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %subview = memref.subview %alloca_3[%arg6, %arg7, 0] [1, 1, %c5] [1, 1, 1] : memref<5x5x5xf64> to memref<?xf64, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map1, #map6], iterator_types = ["reduction", "parallel"]} ins(%subview, %3 : memref<?xf64, strided<[1], offset: ?>>, memref<?x?xf64>) outs(%alloca : memref<4xf64>) {
          ^bb0(%in: f64, %in_4: f64, %out: f64):
            %5 = arith.mulf %in_4, %in : f64
            %6 = arith.addf %out, %5 : f64
            linalg.yield %6 : f64
          }
          %4 = polygeist.submap(%arg1, %arg7, %c4) {map = #map12} : (memref<?xf64>, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map5, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %alloca : memref<?xf64>, memref<4xf64>) outs(%alloca_0 : memref<4x4xf64>) {
          ^bb0(%in: f64, %in_4: f64, %out: f64):
            %5 = arith.mulf %in, %in_4 : f64
            %6 = arith.addf %out, %5 : f64
            linalg.yield %6 : f64
          }
        }
        %1 = polygeist.submap(%arg1, %arg6, %c4, %c4, %c4) {map = #map13} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %2 = polygeist.submap(%arg4, %arg5, %c4, %c4, %c4) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map, #map9, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %alloca_0 : memref<?x?x?xf64>, memref<4x4xf64>) outs(%2 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_4: f64, %out: f64):
          %3 = arith.mulf %in, %in_4 : f64
          %4 = arith.addf %out, %3 : f64
          linalg.yield %4 : f64
        }
      }
    }
    return
  }
}
