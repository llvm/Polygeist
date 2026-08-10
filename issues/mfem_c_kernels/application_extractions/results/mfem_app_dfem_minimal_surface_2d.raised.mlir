#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map9 = affine_map<(d0, d1) -> (d1 + d0 * 5)>
#map10 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 25)>
#map11 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20)>
#map12 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 1)>
#map13 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 2)>
#map14 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 3)>
#map15 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 50)>
#map16 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 75)>
#map17 = affine_map<(d0, d1) -> (d0, d1)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map22 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
    %subview = memref.subview %alloca_3[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %1 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_3 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %subview_4 = memref.subview %alloca_2[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_4 : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2, %3 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_2 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %4 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map5} : (memref<100xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_5 = memref.subview %alloca_2[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %5 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %6 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map5} : (memref<100xf64>, index, index, index) -> memref<2x5x5xf64>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_5, %5 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%6 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %7 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map8} : (memref<100xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_6 = memref.subview %alloca_3[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %8 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %9 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map8} : (memref<100xf64>, index, index, index) -> memref<2x5x5xf64>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_6, %8 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%9 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %10 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map9} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %11 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map10} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %12 = polygeist.submap(%arg3, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %13 = polygeist.submap(%arg3, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %14 = polygeist.submap(%arg3, %c5, %c5) {map = #map13} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %15 = polygeist.submap(%arg3, %c5, %c5) {map = #map14} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %16 = polygeist.submap(%arg4, %c5, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %17 = polygeist.submap(%alloca, %c5, %c5) {map = #map9} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %18 = polygeist.submap(%alloca, %c5, %c5) {map = #map10} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %19 = polygeist.submap(%alloca, %c5, %c5) {map = #map15} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    %20 = polygeist.submap(%alloca, %c5, %c5) {map = #map16} : (memref<100xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"]} ins(%10, %11, %12, %13, %14, %15, %16 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%17, %18, %19, %20 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
    ^bb0(%in: f64, %in_21: f64, %in_22: f64, %in_23: f64, %in_24: f64, %in_25: f64, %in_26: f64, %out: f64, %out_27: f64, %out_28: f64, %out_29: f64):
      %28 = arith.mulf %in_22, %in_25 : f64
      %29 = arith.mulf %in_23, %in_24 : f64
      %30 = arith.subf %28, %29 : f64
      %31 = arith.divf %in_25, %30 : f64
      %32 = arith.negf %in_23 : f64
      %33 = arith.divf %32, %30 : f64
      %34 = arith.negf %in_24 : f64
      %35 = arith.divf %34, %30 : f64
      %36 = arith.divf %in_22, %30 : f64
      %37 = arith.mulf %in, %31 : f64
      %38 = arith.mulf %in_21, %35 : f64
      %39 = arith.addf %37, %38 : f64
      %40 = arith.mulf %in, %33 : f64
      %41 = arith.mulf %in_21, %36 : f64
      %42 = arith.addf %40, %41 : f64
      %43 = arith.mulf %39, %39 : f64
      %44 = arith.addf %43, %cst_0 : f64
      %45 = arith.mulf %42, %42 : f64
      %46 = arith.addf %44, %45 : f64
      %47 = math.sqrt %46 : f64
      %48 = arith.divf %cst_0, %47 : f64
      %49 = arith.mulf %48, %30 : f64
      %50 = arith.mulf %49, %in_26 : f64
      %51 = arith.mulf %39, %31 : f64
      %52 = arith.mulf %42, %33 : f64
      %53 = arith.addf %51, %52 : f64
      %54 = arith.mulf %50, %53 : f64
      %55 = arith.mulf %39, %35 : f64
      %56 = arith.mulf %42, %36 : f64
      %57 = arith.addf %55, %56 : f64
      %58 = arith.mulf %50, %57 : f64
      linalg.yield %54, %58, %cst, %cst : f64, f64, f64, f64
    }
    %alloca_7 = memref.alloca() : memref<2x4x4xf64>
    %alloca_8 = memref.alloca() : memref<2x4x4xf64>
    %alloca_9 = memref.alloca() : memref<2x5x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x4xf64>
    %subview_11 = memref.subview %alloca_10[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_11 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %21 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map18} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %22 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%21, %22 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_10 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %subview_12 = memref.subview %alloca_9[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_12 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %23 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map20} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %24 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%23, %24 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_9 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %subview_13 = memref.subview %alloca_8[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_13 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_14 = memref.subview %alloca_10[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %25 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_15 = memref.subview %alloca_8[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_14, %25 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_15 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %subview_16 = memref.subview %alloca_7[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_16 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_17 = memref.subview %alloca_9[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %26 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_18 = memref.subview %alloca_7[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_17, %26 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_18 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.mulf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    %subview_19 = memref.subview %alloca_8[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %subview_20 = memref.subview %alloca_7[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %27 = polygeist.submap(%arg5, %c2, %c4, %c4) {map = #map22} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_19, %subview_20 : memref<?x?x?xf64, strided<[16, 4, 1]>>, memref<?x?x?xf64, strided<[16, 4, 1]>>) outs(%27 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_21: f64, %out: f64):
      %28 = arith.addf %in, %in_21 : f64
      %29 = arith.addf %out, %28 : f64
      linalg.yield %29 : f64
    }
    return
  }
}

