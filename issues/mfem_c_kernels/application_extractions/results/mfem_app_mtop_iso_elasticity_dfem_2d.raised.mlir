#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50)>
#map9 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4 + 32)>
#map12 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 100)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 100)>
#map14 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 125)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 125)>
#map16 = affine_map<(d0, d1) -> (d1 + d0 * 100)>
#map17 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 25)>
#map18 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 50)>
#map19 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 75)>
#map20 = affine_map<(d0, d1) -> (d1)>
#map21 = affine_map<(d0, d1) -> (d1 + d0 * 25)>
#map22 = affine_map<(d0, d1) -> (d0, d1)>
#map23 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map24 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map25 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map26 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map27 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
#map28 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 100)>
#map29 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 125)>
#map30 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4 + 32)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_mtop_iso_elasticity_dfem_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c25 = arith.constant 25 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<200xf64>
    %alloca_0 = memref.alloca() : memref<200xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5xf64>
    %0 = polygeist.submap(%alloca_2, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %1 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %2 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%alloca_2, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%3 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %4 = polygeist.submap(%alloca_1, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %6 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %7 = polygeist.submap(%alloca_1, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%7 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %8 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map5} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %10 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %11 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4) {map = #map8} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%11 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %12 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map9} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%12 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %13 = polygeist.submap(%alloca_2, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %14 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %15 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4) {map = #map10} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%13, %14 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%15 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %alloca_3 = memref.alloca() : memref<2x4x5xf64>
    %alloca_4 = memref.alloca() : memref<2x4x5xf64>
    %16 = polygeist.submap(%alloca_4, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%16 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %17 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %18 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %19 = polygeist.submap(%alloca_4, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%17, %18 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%19 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %20 = polygeist.submap(%alloca_3, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%20 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %21 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %22 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %23 = polygeist.submap(%alloca_3, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%21, %22 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%23 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %24 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map12} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%24 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %25 = polygeist.submap(%alloca_3, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %26 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %27 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4) {map = #map13} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%25, %26 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%27 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %28 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map14} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%28 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %29 = polygeist.submap(%alloca_4, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %30 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %31 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4) {map = #map15} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%29, %30 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%31 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %32 = polygeist.submap(%arg5, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %33 = polygeist.submap(%arg5, %c2, %c25) {map = #map17} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %34 = polygeist.submap(%arg5, %c2, %c25) {map = #map18} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %35 = polygeist.submap(%arg5, %c2, %c25) {map = #map19} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %36 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map16} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %37 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map17} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %38 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map18} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %39 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map19} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %40 = polygeist.submap(%arg6, %c2, %c25) {map = #map20} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %41 = polygeist.submap(%arg3, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %42 = polygeist.submap(%arg4, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %43 = polygeist.submap(%alloca, %c2, %c25) {map = #map16} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%43 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64):
      %118 = arith.mulf %in, %in_15 : f64
      %119 = arith.mulf %in_13, %in_14 : f64
      %120 = arith.subf %118, %119 : f64
      %121 = arith.divf %in_15, %120 : f64
      %122 = arith.negf %in_13 : f64
      %123 = arith.divf %122, %120 : f64
      %124 = arith.addf %in_16, %in_19 : f64
      %125 = arith.mulf %in_20, %120 : f64
      %126 = arith.mulf %in_21, %121 : f64
      %127 = arith.mulf %126, %124 : f64
      %128 = arith.addf %in_16, %in_16 : f64
      %129 = arith.mulf %121, %128 : f64
      %130 = arith.addf %in_17, %in_18 : f64
      %131 = arith.mulf %123, %130 : f64
      %132 = arith.addf %129, %131 : f64
      %133 = arith.mulf %in_22, %132 : f64
      %134 = arith.addf %127, %133 : f64
      %135 = arith.mulf %125, %134 : f64
      linalg.yield %135 : f64
    }
    %44 = polygeist.submap(%arg5, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %45 = polygeist.submap(%arg5, %c2, %c25) {map = #map17} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %46 = polygeist.submap(%arg5, %c2, %c25) {map = #map18} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %47 = polygeist.submap(%arg5, %c2, %c25) {map = #map19} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %48 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map16} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %49 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map17} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %50 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map18} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %51 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map19} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %52 = polygeist.submap(%arg6, %c2, %c25) {map = #map20} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %53 = polygeist.submap(%arg3, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %54 = polygeist.submap(%arg4, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %55 = polygeist.submap(%alloca, %c2, %c25) {map = #map18} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%55 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64):
      %118 = arith.mulf %in, %in_15 : f64
      %119 = arith.mulf %in_13, %in_14 : f64
      %120 = arith.subf %118, %119 : f64
      %121 = arith.divf %in_15, %120 : f64
      %122 = arith.negf %in_13 : f64
      %123 = arith.divf %122, %120 : f64
      %124 = arith.addf %in_16, %in_19 : f64
      %125 = arith.mulf %in_20, %120 : f64
      %126 = arith.mulf %in_21, %123 : f64
      %127 = arith.mulf %126, %124 : f64
      %128 = arith.addf %in_18, %in_17 : f64
      %129 = arith.mulf %121, %128 : f64
      %130 = arith.addf %in_19, %in_19 : f64
      %131 = arith.mulf %123, %130 : f64
      %132 = arith.addf %129, %131 : f64
      %133 = arith.mulf %in_22, %132 : f64
      %134 = arith.addf %127, %133 : f64
      %135 = arith.mulf %125, %134 : f64
      linalg.yield %135 : f64
    }
    %56 = polygeist.submap(%arg5, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %57 = polygeist.submap(%arg5, %c2, %c25) {map = #map17} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %58 = polygeist.submap(%arg5, %c2, %c25) {map = #map18} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %59 = polygeist.submap(%arg5, %c2, %c25) {map = #map19} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %60 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map16} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %61 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map17} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %62 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map18} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %63 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map19} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %64 = polygeist.submap(%arg6, %c2, %c25) {map = #map20} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %65 = polygeist.submap(%arg3, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %66 = polygeist.submap(%arg4, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %67 = polygeist.submap(%alloca, %c2, %c25) {map = #map17} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%67 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64):
      %118 = arith.mulf %in, %in_15 : f64
      %119 = arith.mulf %in_13, %in_14 : f64
      %120 = arith.subf %118, %119 : f64
      %121 = arith.negf %in_14 : f64
      %122 = arith.divf %121, %120 : f64
      %123 = arith.divf %in, %120 : f64
      %124 = arith.addf %in_16, %in_19 : f64
      %125 = arith.mulf %in_20, %120 : f64
      %126 = arith.mulf %in_21, %122 : f64
      %127 = arith.mulf %126, %124 : f64
      %128 = arith.addf %in_16, %in_16 : f64
      %129 = arith.mulf %122, %128 : f64
      %130 = arith.addf %in_17, %in_18 : f64
      %131 = arith.mulf %123, %130 : f64
      %132 = arith.addf %129, %131 : f64
      %133 = arith.mulf %in_22, %132 : f64
      %134 = arith.addf %127, %133 : f64
      %135 = arith.mulf %125, %134 : f64
      linalg.yield %135 : f64
    }
    %68 = polygeist.submap(%arg5, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %69 = polygeist.submap(%arg5, %c2, %c25) {map = #map17} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %70 = polygeist.submap(%arg5, %c2, %c25) {map = #map18} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %71 = polygeist.submap(%arg5, %c2, %c25) {map = #map19} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %72 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map16} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %73 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map17} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %74 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map18} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %75 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map19} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %76 = polygeist.submap(%arg6, %c2, %c25) {map = #map20} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %77 = polygeist.submap(%arg3, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %78 = polygeist.submap(%arg4, %c2, %c25) {map = #map21} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %79 = polygeist.submap(%alloca, %c2, %c25) {map = #map19} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%79 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %in_22: f64, %out: f64):
      %118 = arith.mulf %in, %in_15 : f64
      %119 = arith.mulf %in_13, %in_14 : f64
      %120 = arith.subf %118, %119 : f64
      %121 = arith.negf %in_14 : f64
      %122 = arith.divf %121, %120 : f64
      %123 = arith.divf %in, %120 : f64
      %124 = arith.addf %in_16, %in_19 : f64
      %125 = arith.mulf %in_20, %120 : f64
      %126 = arith.mulf %in_21, %123 : f64
      %127 = arith.mulf %126, %124 : f64
      %128 = arith.addf %in_18, %in_17 : f64
      %129 = arith.mulf %122, %128 : f64
      %130 = arith.addf %in_19, %in_19 : f64
      %131 = arith.mulf %123, %130 : f64
      %132 = arith.addf %129, %131 : f64
      %133 = arith.mulf %in_22, %132 : f64
      %134 = arith.addf %127, %133 : f64
      %135 = arith.mulf %125, %134 : f64
      linalg.yield %135 : f64
    }
    %alloca_5 = memref.alloca() : memref<2x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x4x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x4xf64>
    %80 = polygeist.submap(%alloca_8, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%80 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %81 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map23} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %82 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map24} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %83 = polygeist.submap(%alloca_8, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%81, %82 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%83 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %84 = polygeist.submap(%alloca_7, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%84 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %85 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map25} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %86 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map24} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %87 = polygeist.submap(%alloca_7, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%85, %86 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%87 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %88 = polygeist.submap(%alloca_6, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%88 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %89 = polygeist.submap(%alloca_8, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %90 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %91 = polygeist.submap(%alloca_6, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%89, %90 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%91 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %92 = polygeist.submap(%alloca_5, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%92 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %93 = polygeist.submap(%alloca_7, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %94 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %95 = polygeist.submap(%alloca_5, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%93, %94 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%95 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %96 = polygeist.submap(%alloca_6, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %97 = polygeist.submap(%alloca_5, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %98 = polygeist.submap(%arg7, %c2, %c4, %c4) {map = #map27} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96, %97 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%98 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.addf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %alloca_9 = memref.alloca() : memref<2x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x4x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x4xf64>
    %99 = polygeist.submap(%alloca_12, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%99 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %100 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map28} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %101 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map24} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %102 = polygeist.submap(%alloca_12, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%100, %101 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%102 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %103 = polygeist.submap(%alloca_11, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%103 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %104 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map29} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %105 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map24} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %106 = polygeist.submap(%alloca_11, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%104, %105 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%106 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %107 = polygeist.submap(%alloca_10, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%107 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %108 = polygeist.submap(%alloca_12, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %109 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %110 = polygeist.submap(%alloca_10, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%108, %109 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%110 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %111 = polygeist.submap(%alloca_9, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%111 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %112 = polygeist.submap(%alloca_11, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %113 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %114 = polygeist.submap(%alloca_9, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%112, %113 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%114 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.mulf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    %115 = polygeist.submap(%alloca_10, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %116 = polygeist.submap(%alloca_9, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %117 = polygeist.submap(%arg7, %c2, %c4, %c4) {map = #map30} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %116 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%117 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_13: f64, %out: f64):
      %118 = arith.addf %in, %in_13 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    }
    return
  }
}
