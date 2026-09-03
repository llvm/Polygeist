#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2 + d0 * 25 + d1 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d2 + d0 * 16 + d1 * 4)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5 + 25)>
#map14 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
#map15 = affine_map<(d0) -> (d0)>
#map16 = affine_map<(d0) -> (0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex9p_mass_convection_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: f64, %arg8: f64, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>, %arg12: memref<?xf64>, %arg13: memref<?xf64>, %arg14: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x5xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5xf64>
    %0 = polygeist.submap(%alloca_1, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %1 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %2 = polygeist.submap(%arg5, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%alloca_1, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%3 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %4 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map} : (memref<2x5x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %6 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %7 = polygeist.submap(%alloca_0, %c2, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%7 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %8 = polygeist.submap(%arg3, %c2, %c5, %c5) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    %9 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map} : (memref<2x5x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : memref<?x?x?xf64>) outs(%9 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %58 = arith.mulf %out, %in : f64
      linalg.yield %58 : f64
    }
    %10 = polygeist.submap(%alloca, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%10 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %11 = polygeist.submap(%arg2, %c2, %c5, %c4, %c5) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %12 = polygeist.submap(%alloca_0, %c2, %c5, %c4, %c5) {map = #map9} : (memref<2x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %13 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%13 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %14 = polygeist.submap(%arg2, %c2, %c4, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %15 = polygeist.submap(%alloca, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %16 = polygeist.submap(%arg9, %c2, %c4, %c4, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%14, %15 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%16 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %alloca_2 = memref.alloca() : memref<2x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x4x5xf64>
    %alloca_7 = memref.alloca() : memref<2x4x5xf64>
    %17 = polygeist.submap(%alloca_7, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%17 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %18 = polygeist.submap(%arg5, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %19 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %20 = polygeist.submap(%alloca_7, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%18, %19 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%20 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %21 = polygeist.submap(%alloca_6, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%21 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %22 = polygeist.submap(%arg5, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %23 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %24 = polygeist.submap(%alloca_6, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%22, %23 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%24 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %25 = polygeist.submap(%alloca_5, %c2, %c5, %c5) {map = #map} : (memref<2x5x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%25 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %26 = polygeist.submap(%alloca_6, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %27 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %28 = polygeist.submap(%alloca_5, %c2, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%26, %27 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%28 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %29 = polygeist.submap(%alloca_4, %c2, %c5, %c5) {map = #map} : (memref<2x5x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%29 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %30 = polygeist.submap(%alloca_7, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %31 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %32 = polygeist.submap(%alloca_4, %c2, %c5, %c5, %c4) {map = #map3} : (memref<2x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%30, %31 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%32 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %33 = polygeist.submap(%alloca_3, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%33 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %34 = polygeist.submap(%arg4, %c2, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %35 = polygeist.submap(%alloca_5, %c2, %c5, %c4, %c5) {map = #map9} : (memref<2x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %36 = polygeist.submap(%arg4, %c2, %c5, %c4, %c5) {map = #map13} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %37 = polygeist.submap(%alloca_4, %c2, %c5, %c4, %c5) {map = #map9} : (memref<2x5x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %38 = polygeist.submap(%arg2, %c2, %c5, %c4, %c5) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %39 = polygeist.submap(%alloca_3, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%34, %35, %36, %37, %38 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%39 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.mulf %in_9, %in_10 : f64
      %60 = arith.addf %58, %59 : f64
      %61 = arith.mulf %60, %in_11 : f64
      %62 = arith.addf %out, %61 : f64
      linalg.yield %62 : f64
    }
    %40 = polygeist.submap(%alloca_2, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%40 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %41 = polygeist.submap(%alloca_3, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %42 = polygeist.submap(%arg2, %c2, %c4, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %43 = polygeist.submap(%alloca_2, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%41, %42 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%43 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %44 = polygeist.submap(%alloca_2, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %45 = polygeist.submap(%arg10, %c2, %c4, %c4) {map = #map14} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44 : memref<?x?x?xf64>) outs(%45 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %58 = arith.addf %out, %in : f64
      linalg.yield %58 : f64
    }
    %46 = polygeist.submap(%arg13, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %47 = polygeist.submap(%arg10, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map15, #map15], iterator_types = ["parallel"]} ins(%46 : memref<?xf64>) outs(%47 : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %58 = arith.mulf %arg7, %in : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %48 = polygeist.submap(%arg9, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %49 = polygeist.submap(%arg11, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map15, #map15], iterator_types = ["parallel"]} ins(%48 : memref<?xf64>) outs(%49 : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %58 = arith.mulf %arg7, %in : f64
      %59 = arith.subf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %50 = polygeist.submap(%arg6, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %51 = polygeist.submap(%arg11, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %52 = polygeist.submap(%arg12, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map15, #map15, #map15], iterator_types = ["parallel"]} ins(%50, %51 : memref<?xf64>, memref<?xf64>) outs(%52 : memref<?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      linalg.yield %58 : f64
    }
    affine.store %cst, %arg14[0] : memref<?xf64>
    %53 = polygeist.submap(%arg11, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %54 = polygeist.submap(%arg12, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %55 = polygeist.submap(%arg14, %c32) {map = #map16} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map15, #map15, #map15], iterator_types = ["reduction"]} ins(%53, %54 : memref<?xf64>, memref<?xf64>) outs(%55 : memref<?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %58 = arith.mulf %in, %in_8 : f64
      %59 = arith.addf %out, %58 : f64
      linalg.yield %59 : f64
    }
    %56 = polygeist.submap(%arg12, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    %57 = polygeist.submap(%arg13, %c32) {map = #map15} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map15, #map15], iterator_types = ["parallel"]} ins(%56 : memref<?xf64>) outs(%57 : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %58 = arith.mulf %arg8, %out : f64
      %59 = arith.addf %in, %58 : f64
      linalg.yield %59 : f64
    }
    return
  }
}
