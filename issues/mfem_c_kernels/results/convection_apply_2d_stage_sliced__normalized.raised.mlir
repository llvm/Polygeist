#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5 + 25)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map12 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x5xf64>
    %alloca_2 = memref.alloca() : memref<2x5x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x5xf64>
    %alloca_4 = memref.alloca() : memref<2x4x5xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_4 : memref<2x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg4, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %1 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_4 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %11 = arith.mulf %in, %in_5 : f64
      %12 = arith.addf %out, %11 : f64
      linalg.yield %12 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_3 : memref<2x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg4, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2, %3 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_3 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %11 = arith.mulf %in, %in_5 : f64
      %12 = arith.addf %out, %11 : f64
      linalg.yield %12 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_2 : memref<2x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %4 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_3, %4 : memref<2x4x5xf64>, memref<?x?x?x?xf64>) outs(%alloca_2 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %11 = arith.mulf %in, %in_5 : f64
      %12 = arith.addf %out, %11 : f64
      linalg.yield %12 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_1 : memref<2x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_4, %5 : memref<2x4x5xf64>, memref<?x?x?x?xf64>) outs(%alloca_1 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %11 = arith.mulf %in, %in_5 : f64
      %12 = arith.addf %out, %11 : f64
      linalg.yield %12 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_0 : memref<2x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %6 = polygeist.submap(%arg3, %c2, %c5, %c4, %c5) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %7 = polygeist.submap(%arg3, %c2, %c5, %c4, %c5) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %8 = polygeist.submap(%arg2, %c2, %c5, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map10, #map3, #map10, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%6, %alloca_2, %7, %alloca_1, %8 : memref<?x?x?x?xf64>, memref<2x5x5xf64>, memref<?x?x?x?xf64>, memref<2x5x5xf64>, memref<?x?x?x?xf64>) outs(%alloca_0 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %11 = arith.mulf %in, %in_5 : f64
      %12 = arith.mulf %in_6, %in_7 : f64
      %13 = arith.addf %11, %12 : f64
      %14 = arith.mulf %13, %in_8 : f64
      %15 = arith.addf %out, %14 : f64
      linalg.yield %15 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca : memref<2x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%arg2, %c2, %c4, %c4, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_0, %9 : memref<2x5x4xf64>, memref<?x?x?x?xf64>) outs(%alloca : memref<2x4x4xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %11 = arith.mulf %in, %in_5 : f64
      %12 = arith.addf %out, %11 : f64
      linalg.yield %12 : f64
    }
    %10 = polygeist.submap(%arg5, %c2, %c4, %c4) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloca : memref<2x4x4xf64>) outs(%10 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %11 = arith.addf %out, %in : f64
      linalg.yield %11 : f64
    }
    return
  }
}
