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
#map11 = affine_map<(d0, d1) -> (d1 + d0 * 5)>
#map12 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 25)>
#map13 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20)>
#map14 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 1)>
#map15 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 2)>
#map16 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 3)>
#map17 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 50)>
#map18 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 75)>
#map19 = affine_map<(d0, d1) -> (d0, d1)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map23 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map24 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_dfem_minimal_surface_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<100xf64>
    %alloca_1 = memref.alloca() : memref<100xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x5xf64>
    %0 = polygeist.submap(%alloca_3, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %1 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %2 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%alloca_3, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%3 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %4 = polygeist.submap(%alloca_2, %c2, %c4, %c5) {map = #map} : (memref<2x4x5xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %6 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %7 = polygeist.submap(%alloca_2, %c2, %c4, %c5, %c4) {map = #map3} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%7 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %8 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map5} : (memref<100xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%alloca_2, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %10 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %11 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4) {map = #map8} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%11 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %12 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map9} : (memref<100xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%12 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %13 = polygeist.submap(%alloca_3, %c2, %c5, %c5, %c4) {map = #map6} : (memref<2x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %14 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %15 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4) {map = #map10} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%13, %14 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%15 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %16 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map11} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %17 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map12} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %18 = polygeist.submap(%arg3, %c5, %c5) {map = #map13} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %19 = polygeist.submap(%arg3, %c5, %c5) {map = #map14} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %20 = polygeist.submap(%arg3, %c5, %c5) {map = #map15} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %21 = polygeist.submap(%arg3, %c5, %c5) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %22 = polygeist.submap(%arg4, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %23 = polygeist.submap(%alloca, %c5, %c5) {map = #map11} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %24 = polygeist.submap(%alloca, %c5, %c5) {map = #map12} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %25 = polygeist.submap(%alloca, %c5, %c5) {map = #map17} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %26 = polygeist.submap(%alloca, %c5, %c5) {map = #map18} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19], iterator_types = ["parallel", "parallel"]} ins(%16, %17, %18, %19, %20, %21, %22 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%23, %24, %25, %26 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %out: f64, %out_14: f64, %out_15: f64, %out_16: f64):
      %46 = arith.mulf %in_9, %in_12 : f64
      %47 = arith.mulf %in_10, %in_11 : f64
      %48 = arith.subf %46, %47 : f64
      %49 = arith.divf %in_12, %48 : f64
      %50 = arith.negf %in_10 : f64
      %51 = arith.divf %50, %48 : f64
      %52 = arith.negf %in_11 : f64
      %53 = arith.divf %52, %48 : f64
      %54 = arith.divf %in_9, %48 : f64
      %55 = arith.mulf %in, %49 : f64
      %56 = arith.mulf %in_8, %53 : f64
      %57 = arith.addf %55, %56 : f64
      %58 = arith.mulf %in, %51 : f64
      %59 = arith.mulf %in_8, %54 : f64
      %60 = arith.addf %58, %59 : f64
      %61 = arith.mulf %57, %57 : f64
      %62 = arith.addf %61, %cst_0 : f64
      %63 = arith.mulf %60, %60 : f64
      %64 = arith.addf %62, %63 : f64
      %65 = math.sqrt %64 : f64
      %66 = arith.divf %cst_0, %65 : f64
      %67 = arith.mulf %66, %48 : f64
      %68 = arith.mulf %67, %in_13 : f64
      %69 = arith.mulf %57, %49 : f64
      %70 = arith.mulf %60, %51 : f64
      %71 = arith.addf %69, %70 : f64
      %72 = arith.mulf %68, %71 : f64
      %73 = arith.mulf %57, %53 : f64
      %74 = arith.mulf %60, %54 : f64
      %75 = arith.addf %73, %74 : f64
      %76 = arith.mulf %68, %75 : f64
      linalg.yield %72, %76, %cst, %cst : f64, f64, f64, f64
    }
    %alloca_4 = memref.alloca() : memref<2x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4xf64>
    %27 = polygeist.submap(%alloca_7, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%27 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %28 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map20} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %29 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %30 = polygeist.submap(%alloca_7, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%28, %29 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%30 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %31 = polygeist.submap(%alloca_6, %c2, %c5, %c4) {map = #map} : (memref<2x5x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%31 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %32 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map22} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %33 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %34 = polygeist.submap(%alloca_6, %c2, %c5, %c4, %c5) {map = #map3} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%32, %33 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%34 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %35 = polygeist.submap(%alloca_5, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%35 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %36 = polygeist.submap(%alloca_7, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %37 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map23} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %38 = polygeist.submap(%alloca_5, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%36, %37 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%38 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %39 = polygeist.submap(%alloca_4, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%39 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %40 = polygeist.submap(%alloca_6, %c2, %c4, %c4, %c5) {map = #map6} : (memref<2x5x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %41 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map23} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %42 = polygeist.submap(%alloca_4, %c2, %c4, %c4, %c5) {map = #map3} : (memref<2x4x4xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%40, %41 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%42 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.mulf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    %43 = polygeist.submap(%alloca_5, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %44 = polygeist.submap(%alloca_4, %c2, %c4, %c4) {map = #map} : (memref<2x4x4xf64>, index, index, index) -> memref<?x?x?xf64>
    %45 = polygeist.submap(%arg5, %c2, %c4, %c4) {map = #map24} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %44 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%45 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %46 = arith.addf %in, %in_8 : f64
      %47 = arith.addf %out, %46 : f64
      linalg.yield %47 : f64
    }
    return
  }
}
