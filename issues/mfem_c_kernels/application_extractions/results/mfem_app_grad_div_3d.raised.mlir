#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 4 + d0 * 108)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 3)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 125 + d1 * 25 + d2 * 5)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 12 + d2 * 4 + d0 * 108)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2, d3)>
#map23 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d1 * 3)>
#map24 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5 + d2 * 3)>
#map25 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 12 + d5 * 4 + d0 * 108)>
#map26 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d3 * 4)>
#map27 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 0, d1, d2, d3)>
#map28 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map29 = affine_map<(d0, d1, d2, d3) -> (d0, 1, d1, d2, d3)>
#map30 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5 + d2 * 4)>
#map31 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 12 + d5 * 3 + d0 * 108 + 36)>
#map32 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d3 * 3)>
#map33 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 1, d1, d2, d3)>
#map34 = affine_map<(d0, d1, d2, d3) -> (d0, 2, d1, d2, d3)>
#map35 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d1 * 4)>
#map36 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 9 + d5 * 3 + d0 * 108 + 72)>
#map37 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 2, d1, d2, d3)>
#map38 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750)>
#map39 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 125)>
#map40 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 250)>
#map41 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 375)>
#map42 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 500)>
#map43 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 625)>
#map44 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d1 * 5)>
#map45 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5 + d2 * 5)>
#map46 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 0, d4, d5, d6)>
#map47 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d3 * 5)>
#map48 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 4 + d0 * 108)>
#map49 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map50 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 1, d4, d5, d6)>
#map51 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map52 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 2, d4, d5, d6)>
#map53 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_grad_div_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x3x3xf64>
    %alloca_0 = memref.alloca() : memref<2x5x4x3xf64>
    %alloca_1 = memref.alloca() : memref<2x5x3x4xf64>
    %alloca_2 = memref.alloca() : memref<2x5x5x3xf64>
    %alloca_3 = memref.alloca() : memref<2x5x5x3xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_7 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_8 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_9 = memref.alloca() : memref<2x3x5x5xf64>
    %alloca_10 = memref.alloca() : memref<2x3x5x5xf64>
    %alloca_11 = memref.alloca() : memref<2x4x3x5xf64>
    %alloca_12 = memref.alloca() : memref<2x3x4x5xf64>
    %alloca_13 = memref.alloca() : memref<2x3x3x5xf64>
    %0 = polygeist.submap(%alloca_13, %c2, %c3, %c3, %c5) {map = #map} : (memref<2x3x3x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %1 = polygeist.submap(%arg8, %c2, %c3, %c3, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %2 = polygeist.submap(%arg4, %c2, %c3, %c3, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %3 = polygeist.submap(%alloca_13, %c2, %c3, %c3, %c5, %c4) {map = #map3} : (memref<2x3x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%3 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %4 = polygeist.submap(%alloca_10, %c2, %c3, %c5, %c5) {map = #map} : (memref<2x3x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%alloca_13, %c2, %c3, %c5, %c5, %c3) {map = #map5} : (memref<2x3x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %6 = polygeist.submap(%arg0, %c2, %c3, %c5, %c5, %c3) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %7 = polygeist.submap(%alloca_10, %c2, %c3, %c5, %c5, %c3) {map = #map3} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%7 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %8 = polygeist.submap(%alloca_7, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%alloca_10, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %10 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3) {map = #map8} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %11 = polygeist.submap(%alloca_7, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%11 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %12 = polygeist.submap(%alloca_12, %c2, %c3, %c4, %c5) {map = #map} : (memref<2x3x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%12 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %13 = polygeist.submap(%arg8, %c2, %c3, %c4, %c5, %c3) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %14 = polygeist.submap(%arg0, %c2, %c3, %c4, %c5, %c3) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %15 = polygeist.submap(%alloca_12, %c2, %c3, %c4, %c5, %c3) {map = #map3} : (memref<2x3x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%13, %14 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%15 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %16 = polygeist.submap(%alloca_9, %c2, %c3, %c5, %c5) {map = #map} : (memref<2x3x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%16 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %17 = polygeist.submap(%alloca_12, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (memref<2x3x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %18 = polygeist.submap(%arg4, %c2, %c3, %c5, %c5, %c4) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %19 = polygeist.submap(%alloca_9, %c2, %c3, %c5, %c5, %c4) {map = #map3} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%17, %18 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%19 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %20 = polygeist.submap(%alloca_6, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%20 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %21 = polygeist.submap(%alloca_9, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %22 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3) {map = #map8} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %23 = polygeist.submap(%alloca_6, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%21, %22 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%23 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %24 = polygeist.submap(%alloca_11, %c2, %c4, %c3, %c5) {map = #map} : (memref<2x4x3x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%24 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %25 = polygeist.submap(%arg8, %c2, %c4, %c3, %c5, %c3) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %26 = polygeist.submap(%arg0, %c2, %c4, %c3, %c5, %c3) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %27 = polygeist.submap(%alloca_11, %c2, %c4, %c3, %c5, %c3) {map = #map3} : (memref<2x4x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%25, %26 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%27 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %28 = polygeist.submap(%alloca_8, %c2, %c4, %c5, %c5) {map = #map} : (memref<2x4x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%28 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %29 = polygeist.submap(%alloca_11, %c2, %c4, %c5, %c5, %c3) {map = #map5} : (memref<2x4x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %30 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c3) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %31 = polygeist.submap(%alloca_8, %c2, %c4, %c5, %c5, %c3) {map = #map3} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%29, %30 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%31 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %32 = polygeist.submap(%alloca_5, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%32 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %33 = polygeist.submap(%alloca_8, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %34 = polygeist.submap(%arg4, %c2, %c5, %c5, %c5, %c4) {map = #map13} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %35 = polygeist.submap(%alloca_5, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%33, %34 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%35 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %36 = polygeist.submap(%alloca_4, %c2, %c5, %c5, %c4) {map = #map} : (memref<2x5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%36 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %37 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %38 = polygeist.submap(%alloca_7, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %39 = polygeist.submap(%alloca_6, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %40 = polygeist.submap(%alloca_5, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %41 = polygeist.submap(%arg5, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %42 = polygeist.submap(%alloca_4, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%37, %38, %39, %40, %41 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%42 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %out: f64):
      %123 = arith.addf %in_15, %in_16 : f64
      %124 = arith.addf %123, %in_17 : f64
      %125 = arith.mulf %in, %124 : f64
      %126 = arith.mulf %125, %in_18 : f64
      %127 = arith.addf %out, %126 : f64
      linalg.yield %127 : f64
    }
    %43 = polygeist.submap(%alloca_3, %c2, %c5, %c5, %c3) {map = #map} : (memref<2x5x5x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%43 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %44 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %45 = polygeist.submap(%alloca_7, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %46 = polygeist.submap(%alloca_6, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %47 = polygeist.submap(%alloca_5, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %48 = polygeist.submap(%arg2, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %49 = polygeist.submap(%alloca_3, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%44, %45, %46, %47, %48 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%49 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %out: f64):
      %123 = arith.addf %in_15, %in_16 : f64
      %124 = arith.addf %123, %in_17 : f64
      %125 = arith.mulf %in, %124 : f64
      %126 = arith.mulf %125, %in_18 : f64
      %127 = arith.addf %out, %126 : f64
      linalg.yield %127 : f64
    }
    %50 = polygeist.submap(%alloca_2, %c2, %c5, %c5, %c3) {map = #map} : (memref<2x5x5x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%50 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %51 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %52 = polygeist.submap(%alloca_7, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %53 = polygeist.submap(%alloca_6, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %54 = polygeist.submap(%alloca_5, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %55 = polygeist.submap(%arg2, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %56 = polygeist.submap(%alloca_2, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%51, %52, %53, %54, %55 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%56 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %out: f64):
      %123 = arith.addf %in_15, %in_16 : f64
      %124 = arith.addf %123, %in_17 : f64
      %125 = arith.mulf %in, %124 : f64
      %126 = arith.mulf %125, %in_18 : f64
      %127 = arith.addf %out, %126 : f64
      linalg.yield %127 : f64
    }
    %57 = polygeist.submap(%alloca_1, %c2, %c5, %c3, %c4) {map = #map} : (memref<2x5x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%57 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %58 = polygeist.submap(%alloca_4, %c2, %c5, %c3, %c4, %c5) {map = #map5} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %59 = polygeist.submap(%arg2, %c2, %c5, %c3, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %60 = polygeist.submap(%alloca_1, %c2, %c5, %c3, %c4, %c5) {map = #map3} : (memref<2x5x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%58, %59 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%60 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %61 = polygeist.submap(%alloca_0, %c2, %c5, %c4, %c3) {map = #map} : (memref<2x5x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%61 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %62 = polygeist.submap(%alloca_3, %c2, %c5, %c4, %c3, %c5) {map = #map5} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %63 = polygeist.submap(%arg5, %c2, %c5, %c4, %c3, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %64 = polygeist.submap(%alloca_0, %c2, %c5, %c4, %c3, %c5) {map = #map3} : (memref<2x5x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%62, %63 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%64 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %65 = polygeist.submap(%alloca, %c2, %c5, %c3, %c3) {map = #map} : (memref<2x5x3x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%65 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %66 = polygeist.submap(%alloca_2, %c2, %c5, %c3, %c3, %c5) {map = #map5} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %67 = polygeist.submap(%arg2, %c2, %c5, %c3, %c3, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %68 = polygeist.submap(%alloca, %c2, %c5, %c3, %c3, %c5) {map = #map3} : (memref<2x5x3x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%66, %67 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%68 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %69 = polygeist.submap(%alloca_1, %c2, %c3, %c3, %c4, %c5) {map = #map7} : (memref<2x5x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %70 = polygeist.submap(%arg2, %c2, %c3, %c3, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %71 = polygeist.submap(%arg9, %c2, %c3, %c3, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%69, %70 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%71 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %72 = polygeist.submap(%alloca_0, %c2, %c3, %c4, %c3, %c5) {map = #map7} : (memref<2x5x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %73 = polygeist.submap(%arg2, %c2, %c3, %c4, %c3, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %74 = polygeist.submap(%arg9, %c2, %c3, %c4, %c3, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%72, %73 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%74 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %75 = polygeist.submap(%alloca, %c2, %c4, %c3, %c3, %c5) {map = #map7} : (memref<2x5x3x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %76 = polygeist.submap(%arg5, %c2, %c4, %c3, %c3, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %77 = polygeist.submap(%arg9, %c2, %c4, %c3, %c3, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%75, %76 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%77 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %out: f64):
      %123 = arith.mulf %in, %in_15 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    }
    %alloca_14 = memref.alloca() : memref<2x3x5x5x5xf64>
    %78 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5) {map = #map22} : (memref<2x3x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%78 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %79 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map23} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %80 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map24} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %81 = polygeist.submap(%arg8, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map25} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %82 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map26} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %83 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map27} : (memref<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map28, #map28, #map28, #map28, #map28], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%79, %80, %81, %82 : memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>) outs(%83 : memref<?x?x?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %123 = arith.mulf %in_16, %in_17 : f64
      %124 = arith.mulf %123, %in_15 : f64
      %125 = arith.mulf %124, %in : f64
      %126 = arith.addf %out, %125 : f64
      linalg.yield %126 : f64
    }
    %84 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5) {map = #map29} : (memref<2x3x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%84 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %85 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map23} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %86 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map30} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %87 = polygeist.submap(%arg8, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map31} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %88 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map32} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %89 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map33} : (memref<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map28, #map28, #map28, #map28, #map28], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%85, %86, %87, %88 : memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>) outs(%89 : memref<?x?x?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %123 = arith.mulf %in_16, %in_17 : f64
      %124 = arith.mulf %123, %in_15 : f64
      %125 = arith.mulf %124, %in : f64
      %126 = arith.addf %out, %125 : f64
      linalg.yield %126 : f64
    }
    %90 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5) {map = #map34} : (memref<2x3x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%90 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %91 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map35} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %92 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map24} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %93 = polygeist.submap(%arg8, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map36} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %94 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map32} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %95 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map37} : (memref<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map28, #map28, #map28, #map28, #map28], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%91, %92, %93, %94 : memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>) outs(%95 : memref<?x?x?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %123 = arith.mulf %in_16, %in_17 : f64
      %124 = arith.mulf %123, %in_15 : f64
      %125 = arith.mulf %124, %in : f64
      %126 = arith.addf %out, %125 : f64
      linalg.yield %126 : f64
    }
    %96 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map38} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %97 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map39} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %98 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map40} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %99 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map39} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %100 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map41} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %101 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map42} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %102 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map40} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %103 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map42} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %104 = polygeist.submap(%arg7, %c2, %c5, %c5, %c5) {map = #map43} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %105 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5) {map = #map22} : (memref<2x3x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %106 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5) {map = #map29} : (memref<2x3x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %107 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c5) {map = #map34} : (memref<2x3x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map, #map, #map, #map, #map, #map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96, %97, %98, %99, %100, %101, %102, %103, %104 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%105, %106, %107 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64, %out_23: f64, %out_24: f64):
      %123 = arith.mulf %in, %out : f64
      %124 = arith.mulf %in_15, %out_23 : f64
      %125 = arith.addf %123, %124 : f64
      %126 = arith.mulf %in_16, %out_24 : f64
      %127 = arith.addf %125, %126 : f64
      %128 = arith.mulf %in_17, %out : f64
      %129 = arith.mulf %in_18, %out_23 : f64
      %130 = arith.addf %128, %129 : f64
      %131 = arith.mulf %in_19, %out_24 : f64
      %132 = arith.addf %130, %131 : f64
      %133 = arith.mulf %in_20, %out : f64
      %134 = arith.mulf %in_21, %out_23 : f64
      %135 = arith.addf %133, %134 : f64
      %136 = arith.mulf %in_22, %out_24 : f64
      %137 = arith.addf %135, %136 : f64
      linalg.yield %127, %132, %137 : f64, f64, f64
    }
    %108 = polygeist.submap(%arg2, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map44} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %109 = polygeist.submap(%arg2, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map45} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %110 = polygeist.submap(%alloca_14, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map46} : (memref<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %111 = polygeist.submap(%arg3, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map47} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %112 = polygeist.submap(%arg9, %c2, %c3, %c3, %c4) {map = #map48} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map28, #map28, #map28, #map28, #map49], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%108, %109, %110, %111 : memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>) outs(%112 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %123 = arith.mulf %in_16, %in_17 : f64
      %124 = arith.mulf %123, %in_15 : f64
      %125 = arith.mulf %124, %in : f64
      %126 = arith.addf %out, %125 : f64
      linalg.yield %126 : f64
    }
    %113 = polygeist.submap(%arg2, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map44} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %114 = polygeist.submap(%arg3, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map45} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %115 = polygeist.submap(%alloca_14, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map50} : (memref<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %116 = polygeist.submap(%arg2, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map47} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %117 = polygeist.submap(%arg9, %c2, %c3, %c4, %c3) {map = #map51} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map28, #map28, #map28, #map28, #map49], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%113, %114, %115, %116 : memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>) outs(%117 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %123 = arith.mulf %in_16, %in_17 : f64
      %124 = arith.mulf %123, %in_15 : f64
      %125 = arith.mulf %124, %in : f64
      %126 = arith.addf %out, %125 : f64
      linalg.yield %126 : f64
    }
    %118 = polygeist.submap(%arg3, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map44} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %119 = polygeist.submap(%arg2, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map45} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %120 = polygeist.submap(%alloca_14, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map52} : (memref<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %121 = polygeist.submap(%arg2, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map47} : (memref<?xf64>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf64>
    %122 = polygeist.submap(%arg9, %c2, %c4, %c3, %c3) {map = #map53} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map28, #map28, #map28, #map28, #map49], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%118, %119, %120, %121 : memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>, memref<?x?x?x?x?x?x?xf64>) outs(%122 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %123 = arith.mulf %in_16, %in_17 : f64
      %124 = arith.mulf %123, %in_15 : f64
      %125 = arith.mulf %124, %in : f64
      %126 = arith.addf %out, %125 : f64
      linalg.yield %126 : f64
    }
    return
  }
}
