#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 3 + d0 * 144)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 4 + d0 * 144 + 48)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 3)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 16 + d2 * 4 + d0 * 144 + 96)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 3)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 125)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 375)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 500)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 250)>
#map22 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 625)>
#map23 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5)>
#map24 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 3 + d0 * 144)>
#map25 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 4 + d0 * 144 + 48)>
#map26 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 16 + d2 * 4 + d0 * 144 + 96)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex35p_hcurl_3d_partial(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_4 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_9 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_13 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_14 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_15 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_16 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_17 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_18 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_21 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_22 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_23 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_24 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_25 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_26 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_27 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_28 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_29 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_30 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_31 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_32 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_33 = memref.alloca() : memref<2x4x4x5xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_33 : memref<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg7, %c2, %c4, %c4, %c5, %c3) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %1 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5, %c3) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_33 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_28 : memref<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg4, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_33, %2 : memref<2x4x4x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_28 : memref<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_27 : memref<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %3 = polygeist.submap(%arg1, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_33, %3 : memref<2x4x4x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_27 : memref<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_22 : memref<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %4 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_28, %4 : memref<2x4x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_22 : memref<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_21 : memref<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_27, %5 : memref<2x4x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_21 : memref<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_32 : memref<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %6 = polygeist.submap(%arg7, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %7 = polygeist.submap(%arg4, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%6, %7 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_32 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_31 : memref<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %8 = polygeist.submap(%arg7, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %9 = polygeist.submap(%arg1, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%8, %9 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_31 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_26 : memref<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %10 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_32, %10 : memref<2x4x4x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_26 : memref<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_25 : memref<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %11 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_31, %11 : memref<2x4x4x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_25 : memref<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_20 : memref<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %12 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_26, %12 : memref<2x4x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_20 : memref<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_19 : memref<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %13 = polygeist.submap(%arg4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_25, %13 : memref<2x4x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_19 : memref<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_30 : memref<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %14 = polygeist.submap(%arg7, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %15 = polygeist.submap(%arg4, %c2, %c3, %c4, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%14, %15 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_30 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_29 : memref<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %16 = polygeist.submap(%arg7, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %17 = polygeist.submap(%arg1, %c2, %c3, %c4, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%16, %17 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_29 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_24 : memref<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %18 = polygeist.submap(%arg1, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_30, %18 : memref<2x4x4x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_24 : memref<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_23 : memref<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %19 = polygeist.submap(%arg4, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_29, %19 : memref<2x4x4x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_23 : memref<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_18 : memref<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %20 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_24, %20 : memref<2x4x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_18 : memref<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_17 : memref<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %21 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_23, %21 : memref<2x4x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_17 : memref<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_16 : memref<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %22 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %23 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %24 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %25 = polygeist.submap(%arg2, %c2, %c5, %c5, %c3, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%22, %alloca_17, %alloca_19, %23, %alloca_21, %alloca_18, %24, %alloca_20, %alloca_22, %25 : memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_16 : memref<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %61 = arith.subf %in_34, %in_35 : f64
      %62 = arith.mulf %in, %61 : f64
      %63 = arith.subf %in_37, %in_38 : f64
      %64 = arith.mulf %in_36, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %66 = arith.subf %in_40, %in_41 : f64
      %67 = arith.mulf %in_39, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %69 = arith.mulf %68, %in_42 : f64
      %70 = arith.addf %out, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_10 : memref<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %26 = polygeist.submap(%arg3, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_16, %26 : memref<2x5x5x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_10 : memref<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_4 : memref<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %27 = polygeist.submap(%arg5, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_10, %27 : memref<2x5x4x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_4 : memref<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_15 : memref<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %28 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %29 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %30 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %31 = polygeist.submap(%arg2, %c2, %c5, %c5, %c3, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%28, %alloca_17, %alloca_19, %29, %alloca_21, %alloca_18, %30, %alloca_20, %alloca_22, %31 : memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_15 : memref<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %61 = arith.subf %in_34, %in_35 : f64
      %62 = arith.mulf %in, %61 : f64
      %63 = arith.subf %in_37, %in_38 : f64
      %64 = arith.mulf %in_36, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %66 = arith.subf %in_40, %in_41 : f64
      %67 = arith.mulf %in_39, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %69 = arith.mulf %68, %in_42 : f64
      %70 = arith.addf %out, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_9 : memref<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %32 = polygeist.submap(%arg5, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_15, %32 : memref<2x5x5x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_9 : memref<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_3 : memref<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %33 = polygeist.submap(%arg3, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_9, %33 : memref<2x5x4x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_3 : memref<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_14 : memref<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %34 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %35 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %36 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %37 = polygeist.submap(%arg5, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%34, %alloca_17, %alloca_19, %35, %alloca_21, %alloca_18, %36, %alloca_20, %alloca_22, %37 : memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_14 : memref<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %61 = arith.subf %in_34, %in_35 : f64
      %62 = arith.mulf %in, %61 : f64
      %63 = arith.subf %in_37, %in_38 : f64
      %64 = arith.mulf %in_36, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %66 = arith.subf %in_40, %in_41 : f64
      %67 = arith.mulf %in_39, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %69 = arith.mulf %68, %in_42 : f64
      %70 = arith.addf %out, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_8 : memref<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %38 = polygeist.submap(%arg2, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_14, %38 : memref<2x5x5x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_8 : memref<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_2 : memref<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %39 = polygeist.submap(%arg3, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_8, %39 : memref<2x5x4x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_2 : memref<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_13 : memref<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %40 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map23} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %41 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %42 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %43 = polygeist.submap(%arg3, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%40, %alloca_17, %alloca_19, %41, %alloca_21, %alloca_18, %42, %alloca_20, %alloca_22, %43 : memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_13 : memref<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %61 = arith.subf %in_34, %in_35 : f64
      %62 = arith.mulf %in, %61 : f64
      %63 = arith.subf %in_37, %in_38 : f64
      %64 = arith.mulf %in_36, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %66 = arith.subf %in_40, %in_41 : f64
      %67 = arith.mulf %in_39, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %69 = arith.mulf %68, %in_42 : f64
      %70 = arith.addf %out, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_7 : memref<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %44 = polygeist.submap(%arg2, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_13, %44 : memref<2x5x5x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_7 : memref<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_1 : memref<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %45 = polygeist.submap(%arg5, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_7, %45 : memref<2x5x4x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_1 : memref<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_12 : memref<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %46 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map23} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %47 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %48 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %49 = polygeist.submap(%arg3, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%46, %alloca_17, %alloca_19, %47, %alloca_21, %alloca_18, %48, %alloca_20, %alloca_22, %49 : memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_12 : memref<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %61 = arith.subf %in_34, %in_35 : f64
      %62 = arith.mulf %in, %61 : f64
      %63 = arith.subf %in_37, %in_38 : f64
      %64 = arith.mulf %in_36, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %66 = arith.subf %in_40, %in_41 : f64
      %67 = arith.mulf %in_39, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %69 = arith.mulf %68, %in_42 : f64
      %70 = arith.addf %out, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_6 : memref<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %50 = polygeist.submap(%arg5, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_12, %50 : memref<2x5x5x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_6 : memref<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_0 : memref<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %51 = polygeist.submap(%arg2, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_6, %51 : memref<2x5x4x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_0 : memref<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_11 : memref<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %52 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %53 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %54 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %55 = polygeist.submap(%arg5, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%52, %alloca_17, %alloca_19, %53, %alloca_21, %alloca_18, %54, %alloca_20, %alloca_22, %55 : memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>, memref<2x5x5x5xf64>, memref<2x5x5x5xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_11 : memref<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %61 = arith.subf %in_34, %in_35 : f64
      %62 = arith.mulf %in, %61 : f64
      %63 = arith.subf %in_37, %in_38 : f64
      %64 = arith.mulf %in_36, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %66 = arith.subf %in_40, %in_41 : f64
      %67 = arith.mulf %in_39, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %69 = arith.mulf %68, %in_42 : f64
      %70 = arith.addf %out, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_5 : memref<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %56 = polygeist.submap(%arg3, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_11, %56 : memref<2x5x5x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_5 : memref<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca : memref<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %57 = polygeist.submap(%arg2, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%alloca_5, %57 : memref<2x5x4x4xf64>, memref<?x?x?x?x?xf64>) outs(%alloca : memref<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.mulf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    %58 = polygeist.submap(%arg8, %c2, %c4, %c4, %c3) {map = #map24} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloca_4, %alloca_3 : memref<2x4x4x4xf64>, memref<2x4x4x4xf64>) outs(%58 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.subf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    %59 = polygeist.submap(%arg8, %c2, %c4, %c3, %c4) {map = #map25} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloca_2, %alloca_1 : memref<2x4x4x4xf64>, memref<2x4x4x4xf64>) outs(%59 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.subf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    %60 = polygeist.submap(%arg8, %c2, %c3, %c4, %c4) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloca_0, %alloca : memref<2x4x4x4xf64>, memref<2x4x4x4xf64>) outs(%60 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %61 = arith.subf %in, %in_34 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    return
  }
}
