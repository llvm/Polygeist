#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 125 + d1 * 25 + d2 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_mass_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x5xf64>
    %0 = polygeist.submap(%alloca_3, %c2, %c4, %c4, %c5) {map = #map} : (memref<2x4x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %1 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %2 = polygeist.submap(%arg3, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %3 = polygeist.submap(%alloca_3, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (memref<2x4x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%3 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %25 = arith.mulf %in, %in_4 : f64
      %26 = arith.addf %out, %25 : f64
      linalg.yield %26 : f64
    }
    %4 = polygeist.submap(%alloca_2, %c2, %c4, %c5, %c5) {map = #map} : (memref<2x4x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %6 = polygeist.submap(%alloca_3, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (memref<2x4x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %7 = polygeist.submap(%alloca_2, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%7 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %25 = arith.mulf %in, %in_4 : f64
      %26 = arith.addf %out, %25 : f64
      linalg.yield %26 : f64
    }
    %8 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %10 = polygeist.submap(%alloca_2, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %11 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%11 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %25 = arith.mulf %in, %in_4 : f64
      %26 = arith.addf %out, %25 : f64
      linalg.yield %26 : f64
    }
    %12 = polygeist.submap(%arg2, %c2, %c5, %c5, %c5) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %13 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : memref<?x?x?x?xf64>) outs(%13 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %25 = arith.mulf %out, %in : f64
      linalg.yield %25 : f64
    }
    %14 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4) {map = #map} : (memref<2x5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%14 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %15 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %16 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %17 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%15, %16 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%17 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %25 = arith.mulf %in, %in_4 : f64
      %26 = arith.addf %out, %25 : f64
      linalg.yield %26 : f64
    }
    %18 = polygeist.submap(%alloca, %c2, %c5, %c4, %c4) {map = #map} : (memref<2x5x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%18 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %19 = polygeist.submap(%arg1, %c2, %c5, %c4, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %20 = polygeist.submap(%alloca_0, %c2, %c5, %c4, %c4, %c5) {map = #map6} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %21 = polygeist.submap(%alloca, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (memref<2x5x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%19, %20 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%21 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %25 = arith.mulf %in, %in_4 : f64
      %26 = arith.addf %out, %25 : f64
      linalg.yield %26 : f64
    }
    %22 = polygeist.submap(%arg1, %c2, %c4, %c4, %c4, %c5) {map = #map13} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %23 = polygeist.submap(%alloca, %c2, %c4, %c4, %c4, %c5) {map = #map8} : (memref<2x5x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %24 = polygeist.submap(%arg4, %c2, %c4, %c4, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%22, %23 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%24 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %25 = arith.mulf %in, %in_4 : f64
      %26 = arith.addf %out, %25 : f64
      linalg.yield %26 : f64
    }
    return
  }
}
