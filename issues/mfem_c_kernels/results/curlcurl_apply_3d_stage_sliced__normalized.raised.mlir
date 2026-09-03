#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 3 + d0 * 144)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 4 + d0 * 144 + 48)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 3)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 16 + d2 * 4 + d0 * 144 + 96)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 3)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 125)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 375)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 500)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 250)>
#map22 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 625)>
#map23 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5)>
#map24 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 3 + d0 * 144)>
#map25 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 4 + d0 * 144 + 48)>
#map26 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 16 + d2 * 4 + d0 * 144 + 96)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_curlcurl_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x3x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x3x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x4x3x4xf64>
    %alloca_2 = memref.alloca() : memref<2x4x3x4xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x3xf64>
    %alloca_4 = memref.alloca() : memref<2x4x4x3xf64>
    %alloca_5 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x3x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x3x4xf64>
    %alloca_9 = memref.alloca() : memref<2x5x4x3xf64>
    %alloca_10 = memref.alloca() : memref<2x5x4x3xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_13 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_14 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_15 = memref.alloca() : memref<2x5x5x3xf64>
    %alloca_16 = memref.alloca() : memref<2x5x5x3xf64>
    %alloca_17 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_18 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_21 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_22 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_23 = memref.alloca() : memref<2x3x5x5xf64>
    %alloca_24 = memref.alloca() : memref<2x3x5x5xf64>
    %alloca_25 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_26 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_27 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_28 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_29 = memref.alloca() : memref<2x3x4x5xf64>
    %alloca_30 = memref.alloca() : memref<2x3x4x5xf64>
    %alloca_31 = memref.alloca() : memref<2x4x3x5xf64>
    %alloca_32 = memref.alloca() : memref<2x4x3x5xf64>
    %alloca_33 = memref.alloca() : memref<2x4x4x5xf64>
    %0 = polygeist.submap(%alloca_33, %c2, %c4, %c4, %c5) {map = #map} : (memref<2x4x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %1 = polygeist.submap(%arg7, %c2, %c4, %c4, %c5, %c3) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %2 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5, %c3) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %3 = polygeist.submap(%alloca_33, %c2, %c4, %c4, %c5, %c3) {map = #map3} : (memref<2x4x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%3 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %4 = polygeist.submap(%alloca_28, %c2, %c4, %c5, %c5) {map = #map} : (memref<2x4x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%alloca_33, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<2x4x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %6 = polygeist.submap(%arg4, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %7 = polygeist.submap(%alloca_28, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%7 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %8 = polygeist.submap(%alloca_27, %c2, %c4, %c5, %c5) {map = #map} : (memref<2x4x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%alloca_33, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<2x4x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %10 = polygeist.submap(%arg1, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %11 = polygeist.submap(%alloca_27, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%11 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %12 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%12 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %13 = polygeist.submap(%alloca_28, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %14 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %15 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%13, %14 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%15 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %16 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%16 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %17 = polygeist.submap(%alloca_27, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %18 = polygeist.submap(%arg4, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %19 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%17, %18 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%19 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %20 = polygeist.submap(%alloca_32, %c2, %c4, %c3, %c5) {map = #map} : (memref<2x4x3x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%20 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %21 = polygeist.submap(%arg7, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %22 = polygeist.submap(%arg4, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %23 = polygeist.submap(%alloca_32, %c2, %c4, %c3, %c5, %c4) {map = #map3} : (memref<2x4x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%21, %22 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%23 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %24 = polygeist.submap(%alloca_31, %c2, %c4, %c3, %c5) {map = #map} : (memref<2x4x3x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%24 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %25 = polygeist.submap(%arg7, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %26 = polygeist.submap(%arg1, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %27 = polygeist.submap(%alloca_31, %c2, %c4, %c3, %c5, %c4) {map = #map3} : (memref<2x4x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%25, %26 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%27 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %28 = polygeist.submap(%alloca_26, %c2, %c4, %c5, %c5) {map = #map} : (memref<2x4x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%28 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %29 = polygeist.submap(%alloca_32, %c2, %c4, %c5, %c5, %c3) {map = #map5} : (memref<2x4x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %30 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %31 = polygeist.submap(%alloca_26, %c2, %c4, %c5, %c5, %c3) {map = #map3} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%29, %30 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%31 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %32 = polygeist.submap(%alloca_25, %c2, %c4, %c5, %c5) {map = #map} : (memref<2x4x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%32 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %33 = polygeist.submap(%alloca_31, %c2, %c4, %c5, %c5, %c3) {map = #map5} : (memref<2x4x3x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %34 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %35 = polygeist.submap(%alloca_25, %c2, %c4, %c5, %c5, %c3) {map = #map3} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%33, %34 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%35 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %36 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%36 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %37 = polygeist.submap(%alloca_26, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %38 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %39 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%37, %38 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%39 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %40 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%40 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %41 = polygeist.submap(%alloca_25, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<2x4x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %42 = polygeist.submap(%arg4, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %43 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%41, %42 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%43 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %44 = polygeist.submap(%alloca_30, %c2, %c3, %c4, %c5) {map = #map} : (memref<2x3x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%44 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %45 = polygeist.submap(%arg7, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %46 = polygeist.submap(%arg4, %c2, %c3, %c4, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %47 = polygeist.submap(%alloca_30, %c2, %c3, %c4, %c5, %c4) {map = #map3} : (memref<2x3x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%45, %46 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%47 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %48 = polygeist.submap(%alloca_29, %c2, %c3, %c4, %c5) {map = #map} : (memref<2x3x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%48 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %49 = polygeist.submap(%arg7, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %50 = polygeist.submap(%arg1, %c2, %c3, %c4, %c5, %c4) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %51 = polygeist.submap(%alloca_29, %c2, %c3, %c4, %c5, %c4) {map = #map3} : (memref<2x3x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%49, %50 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%51 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %52 = polygeist.submap(%alloca_24, %c2, %c3, %c5, %c5) {map = #map} : (memref<2x3x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%52 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %53 = polygeist.submap(%alloca_30, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (memref<2x3x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %54 = polygeist.submap(%arg1, %c2, %c3, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %55 = polygeist.submap(%alloca_24, %c2, %c3, %c5, %c5, %c4) {map = #map3} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%53, %54 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%55 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %56 = polygeist.submap(%alloca_23, %c2, %c3, %c5, %c5) {map = #map} : (memref<2x3x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%56 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %57 = polygeist.submap(%alloca_29, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (memref<2x3x4x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %58 = polygeist.submap(%arg4, %c2, %c3, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %59 = polygeist.submap(%alloca_23, %c2, %c3, %c5, %c5, %c4) {map = #map3} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%57, %58 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%59 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %60 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%60 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %61 = polygeist.submap(%alloca_24, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %62 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %63 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%61, %62 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%63 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %64 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c5) {map = #map} : (memref<2x5x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%64 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %65 = polygeist.submap(%alloca_23, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (memref<2x3x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %66 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %67 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%65, %66 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%67 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %68 = polygeist.submap(%alloca_16, %c2, %c5, %c5, %c3) {map = #map} : (memref<2x5x5x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%68 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %69 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %70 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %71 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %72 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %73 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %74 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %75 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %76 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %77 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %78 = polygeist.submap(%arg2, %c2, %c5, %c5, %c3, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %79 = polygeist.submap(%alloca_16, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%69, %70, %71, %72, %73, %74, %75, %76, %77, %78 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%79 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %197 = arith.subf %in_34, %in_35 : f64
      %198 = arith.mulf %in, %197 : f64
      %199 = arith.subf %in_37, %in_38 : f64
      %200 = arith.mulf %in_36, %199 : f64
      %201 = arith.addf %198, %200 : f64
      %202 = arith.subf %in_40, %in_41 : f64
      %203 = arith.mulf %in_39, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.mulf %204, %in_42 : f64
      %206 = arith.addf %out, %205 : f64
      linalg.yield %206 : f64
    }
    %80 = polygeist.submap(%alloca_10, %c2, %c5, %c4, %c3) {map = #map} : (memref<2x5x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%80 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %81 = polygeist.submap(%alloca_16, %c2, %c5, %c4, %c3, %c5) {map = #map5} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %82 = polygeist.submap(%arg3, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %83 = polygeist.submap(%alloca_10, %c2, %c5, %c4, %c3, %c5) {map = #map3} : (memref<2x5x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%81, %82 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%83 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %84 = polygeist.submap(%alloca_4, %c2, %c4, %c4, %c3) {map = #map} : (memref<2x4x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%84 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %85 = polygeist.submap(%alloca_10, %c2, %c4, %c4, %c3, %c5) {map = #map7} : (memref<2x5x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %86 = polygeist.submap(%arg5, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %87 = polygeist.submap(%alloca_4, %c2, %c4, %c4, %c3, %c5) {map = #map3} : (memref<2x4x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%85, %86 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%87 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %88 = polygeist.submap(%alloca_15, %c2, %c5, %c5, %c3) {map = #map} : (memref<2x5x5x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%88 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %89 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %90 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %91 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %92 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %93 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %94 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %95 = polygeist.submap(%arg6, %c2, %c5, %c5, %c3, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %96 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %97 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %98 = polygeist.submap(%arg2, %c2, %c5, %c5, %c3, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %99 = polygeist.submap(%alloca_15, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%89, %90, %91, %92, %93, %94, %95, %96, %97, %98 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%99 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %197 = arith.subf %in_34, %in_35 : f64
      %198 = arith.mulf %in, %197 : f64
      %199 = arith.subf %in_37, %in_38 : f64
      %200 = arith.mulf %in_36, %199 : f64
      %201 = arith.addf %198, %200 : f64
      %202 = arith.subf %in_40, %in_41 : f64
      %203 = arith.mulf %in_39, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.mulf %204, %in_42 : f64
      %206 = arith.addf %out, %205 : f64
      linalg.yield %206 : f64
    }
    %100 = polygeist.submap(%alloca_9, %c2, %c5, %c4, %c3) {map = #map} : (memref<2x5x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%100 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %101 = polygeist.submap(%alloca_15, %c2, %c5, %c4, %c3, %c5) {map = #map5} : (memref<2x5x5x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %102 = polygeist.submap(%arg5, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %103 = polygeist.submap(%alloca_9, %c2, %c5, %c4, %c3, %c5) {map = #map3} : (memref<2x5x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%101, %102 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%103 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %104 = polygeist.submap(%alloca_3, %c2, %c4, %c4, %c3) {map = #map} : (memref<2x4x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%104 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %105 = polygeist.submap(%alloca_9, %c2, %c4, %c4, %c3, %c5) {map = #map7} : (memref<2x5x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %106 = polygeist.submap(%arg3, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %107 = polygeist.submap(%alloca_3, %c2, %c4, %c4, %c3, %c5) {map = #map3} : (memref<2x4x4x3xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%105, %106 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%107 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %108 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c4) {map = #map} : (memref<2x5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%108 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %109 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %110 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %111 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %112 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %113 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %114 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %115 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %116 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %117 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %118 = polygeist.submap(%arg5, %c2, %c5, %c5, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %119 = polygeist.submap(%alloca_14, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%109, %110, %111, %112, %113, %114, %115, %116, %117, %118 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%119 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %197 = arith.subf %in_34, %in_35 : f64
      %198 = arith.mulf %in, %197 : f64
      %199 = arith.subf %in_37, %in_38 : f64
      %200 = arith.mulf %in_36, %199 : f64
      %201 = arith.addf %198, %200 : f64
      %202 = arith.subf %in_40, %in_41 : f64
      %203 = arith.mulf %in_39, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.mulf %204, %in_42 : f64
      %206 = arith.addf %out, %205 : f64
      linalg.yield %206 : f64
    }
    %120 = polygeist.submap(%alloca_8, %c2, %c5, %c3, %c4) {map = #map} : (memref<2x5x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%120 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %121 = polygeist.submap(%alloca_14, %c2, %c5, %c3, %c4, %c5) {map = #map5} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %122 = polygeist.submap(%arg2, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %123 = polygeist.submap(%alloca_8, %c2, %c5, %c3, %c4, %c5) {map = #map3} : (memref<2x5x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%121, %122 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%123 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %124 = polygeist.submap(%alloca_2, %c2, %c4, %c3, %c4) {map = #map} : (memref<2x4x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%124 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %125 = polygeist.submap(%alloca_8, %c2, %c4, %c3, %c4, %c5) {map = #map7} : (memref<2x5x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %126 = polygeist.submap(%arg3, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %127 = polygeist.submap(%alloca_2, %c2, %c4, %c3, %c4, %c5) {map = #map3} : (memref<2x4x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%125, %126 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%127 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %128 = polygeist.submap(%alloca_13, %c2, %c5, %c5, %c4) {map = #map} : (memref<2x5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%128 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %129 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map23} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %130 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %131 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %132 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %133 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %134 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %135 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %136 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %137 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %138 = polygeist.submap(%arg3, %c2, %c5, %c5, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %139 = polygeist.submap(%alloca_13, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%129, %130, %131, %132, %133, %134, %135, %136, %137, %138 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%139 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %197 = arith.subf %in_34, %in_35 : f64
      %198 = arith.mulf %in, %197 : f64
      %199 = arith.subf %in_37, %in_38 : f64
      %200 = arith.mulf %in_36, %199 : f64
      %201 = arith.addf %198, %200 : f64
      %202 = arith.subf %in_40, %in_41 : f64
      %203 = arith.mulf %in_39, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.mulf %204, %in_42 : f64
      %206 = arith.addf %out, %205 : f64
      linalg.yield %206 : f64
    }
    %140 = polygeist.submap(%alloca_7, %c2, %c5, %c3, %c4) {map = #map} : (memref<2x5x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%140 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %141 = polygeist.submap(%alloca_13, %c2, %c5, %c3, %c4, %c5) {map = #map5} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %142 = polygeist.submap(%arg2, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %143 = polygeist.submap(%alloca_7, %c2, %c5, %c3, %c4, %c5) {map = #map3} : (memref<2x5x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%141, %142 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%143 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %144 = polygeist.submap(%alloca_1, %c2, %c4, %c3, %c4) {map = #map} : (memref<2x4x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%144 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %145 = polygeist.submap(%alloca_7, %c2, %c4, %c3, %c4, %c5) {map = #map7} : (memref<2x5x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %146 = polygeist.submap(%arg5, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %147 = polygeist.submap(%alloca_1, %c2, %c4, %c3, %c4, %c5) {map = #map3} : (memref<2x4x3x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%145, %146 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%147 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %148 = polygeist.submap(%alloca_12, %c2, %c5, %c5, %c4) {map = #map} : (memref<2x5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%148 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %149 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map23} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %150 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %151 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %152 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %153 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %154 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %155 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %156 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %157 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %158 = polygeist.submap(%arg3, %c2, %c5, %c5, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %159 = polygeist.submap(%alloca_12, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%149, %150, %151, %152, %153, %154, %155, %156, %157, %158 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%159 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %197 = arith.subf %in_34, %in_35 : f64
      %198 = arith.mulf %in, %197 : f64
      %199 = arith.subf %in_37, %in_38 : f64
      %200 = arith.mulf %in_36, %199 : f64
      %201 = arith.addf %198, %200 : f64
      %202 = arith.subf %in_40, %in_41 : f64
      %203 = arith.mulf %in_39, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.mulf %204, %in_42 : f64
      %206 = arith.addf %out, %205 : f64
      linalg.yield %206 : f64
    }
    %160 = polygeist.submap(%alloca_6, %c2, %c5, %c4, %c4) {map = #map} : (memref<2x5x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%160 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %161 = polygeist.submap(%alloca_12, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %162 = polygeist.submap(%arg5, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %163 = polygeist.submap(%alloca_6, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (memref<2x5x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%161, %162 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%163 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %164 = polygeist.submap(%alloca_0, %c2, %c3, %c4, %c4) {map = #map} : (memref<2x3x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%164 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %165 = polygeist.submap(%alloca_6, %c2, %c3, %c4, %c4, %c5) {map = #map7} : (memref<2x5x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %166 = polygeist.submap(%arg2, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %167 = polygeist.submap(%alloca_0, %c2, %c3, %c4, %c4, %c5) {map = #map3} : (memref<2x3x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%165, %166 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%167 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %168 = polygeist.submap(%alloca_11, %c2, %c5, %c5, %c4) {map = #map} : (memref<2x5x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%168 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %169 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %170 = polygeist.submap(%alloca_17, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %171 = polygeist.submap(%alloca_19, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %172 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %173 = polygeist.submap(%alloca_21, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %174 = polygeist.submap(%alloca_18, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %175 = polygeist.submap(%arg6, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %176 = polygeist.submap(%alloca_20, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %177 = polygeist.submap(%alloca_22, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<2x5x5x5xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %178 = polygeist.submap(%arg5, %c2, %c5, %c5, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %179 = polygeist.submap(%alloca_11, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%169, %170, %171, %172, %173, %174, %175, %176, %177, %178 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%179 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64):
      %197 = arith.subf %in_34, %in_35 : f64
      %198 = arith.mulf %in, %197 : f64
      %199 = arith.subf %in_37, %in_38 : f64
      %200 = arith.mulf %in_36, %199 : f64
      %201 = arith.addf %198, %200 : f64
      %202 = arith.subf %in_40, %in_41 : f64
      %203 = arith.mulf %in_39, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.mulf %204, %in_42 : f64
      %206 = arith.addf %out, %205 : f64
      linalg.yield %206 : f64
    }
    %180 = polygeist.submap(%alloca_5, %c2, %c5, %c4, %c4) {map = #map} : (memref<2x5x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%180 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %181 = polygeist.submap(%alloca_11, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (memref<2x5x5x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %182 = polygeist.submap(%arg3, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %183 = polygeist.submap(%alloca_5, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (memref<2x5x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%181, %182 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%183 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %184 = polygeist.submap(%alloca, %c2, %c3, %c4, %c4) {map = #map} : (memref<2x3x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%184 : memref<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %185 = polygeist.submap(%alloca_5, %c2, %c3, %c4, %c4, %c5) {map = #map7} : (memref<2x5x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %186 = polygeist.submap(%arg2, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %187 = polygeist.submap(%alloca, %c2, %c3, %c4, %c4, %c5) {map = #map3} : (memref<2x3x4x4xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%185, %186 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%187 : memref<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.mulf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %188 = polygeist.submap(%alloca_4, %c2, %c4, %c4, %c3) {map = #map} : (memref<2x4x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %189 = polygeist.submap(%alloca_3, %c2, %c4, %c4, %c3) {map = #map} : (memref<2x4x4x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %190 = polygeist.submap(%arg8, %c2, %c4, %c4, %c3) {map = #map24} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%188, %189 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%190 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.subf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %191 = polygeist.submap(%alloca_2, %c2, %c4, %c3, %c4) {map = #map} : (memref<2x4x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %192 = polygeist.submap(%alloca_1, %c2, %c4, %c3, %c4) {map = #map} : (memref<2x4x3x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %193 = polygeist.submap(%arg8, %c2, %c4, %c3, %c4) {map = #map25} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%191, %192 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%193 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.subf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    %194 = polygeist.submap(%alloca_0, %c2, %c3, %c4, %c4) {map = #map} : (memref<2x3x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %195 = polygeist.submap(%alloca, %c2, %c3, %c4, %c4) {map = #map} : (memref<2x3x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %196 = polygeist.submap(%arg8, %c2, %c3, %c4, %c4) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%194, %195 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%196 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_34: f64, %out: f64):
      %197 = arith.subf %in, %in_34 : f64
      %198 = arith.addf %out, %197 : f64
      linalg.yield %198 : f64
    }
    return
  }
}
