#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d2 * 5 + d0 * 750)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 125)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 250)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 375)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 500)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 625)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_diffusion_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_9 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_10 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_11 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_13 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_14 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_15 = memref.alloca() : memref<2x4x4x5xf64>
    %subview = memref.subview %alloca_15[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : memref<2x4x4x5xf64> to memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %1 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_15 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_16 = memref.subview %alloca_14[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : memref<2x4x4x5xf64> to memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_16 : memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %3 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2, %3 : memref<?x?x?x?x?xf64>, memref<?x?x?x?x?xf64>) outs(%alloca_14 : memref<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_17 = memref.subview %alloca_13[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_17 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_18 = memref.subview %alloca_14[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : memref<2x4x4x5xf64> to memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>
    %4 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_19 = memref.subview %alloca_13[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_18, %4 : memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_19 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_20 = memref.subview %alloca_12[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_20 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_21 = memref.subview %alloca_15[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : memref<2x4x4x5xf64> to memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>
    %5 = polygeist.submap(%arg1, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_22 = memref.subview %alloca_12[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_21, %5 : memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_22 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_23 = memref.subview %alloca_11[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_23 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_24 = memref.subview %alloca_15[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : memref<2x4x4x5xf64> to memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>
    %6 = polygeist.submap(%arg0, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_25 = memref.subview %alloca_11[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_24, %6 : memref<?x?x?x?xf64, strided<[80, 20, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_25 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_26 = memref.subview %alloca_10[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_26 : memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_27 = memref.subview %alloca_13[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    %7 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_28 = memref.subview %alloca_10[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_27, %7 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_28 : memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_29 = memref.subview %alloca_9[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_29 : memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_30 = memref.subview %alloca_12[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    %8 = polygeist.submap(%arg0, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_31 = memref.subview %alloca_9[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_30, %8 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_31 : memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_32 = memref.subview %alloca_8[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_32 : memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_33 = memref.subview %alloca_11[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : memref<2x4x5x5xf64> to memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>
    %9 = polygeist.submap(%arg1, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_34 = memref.subview %alloca_8[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_33, %9 : memref<?x?x?x?xf64, strided<[100, 25, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_34 : memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_35 = memref.subview %alloca_7[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_35 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %10 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_36 = memref.subview %alloca_10[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %11 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_37 = memref.subview %alloca_9[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %12 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_38 = memref.subview %alloca_8[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %13 = polygeist.submap(%arg3, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_39 = memref.subview %alloca_7[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    linalg.generic {indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%10, %subview_36, %11, %subview_37, %12, %subview_38, %13 : memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_39 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %in_72: f64, %in_73: f64, %in_74: f64, %in_75: f64, %in_76: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.mulf %in_72, %in_73 : f64
      %31 = arith.addf %29, %30 : f64
      %32 = arith.mulf %in_74, %in_75 : f64
      %33 = arith.addf %31, %32 : f64
      %34 = arith.mulf %33, %in_76 : f64
      %35 = arith.addf %out, %34 : f64
      linalg.yield %35 : f64
    }
    %subview_40 = memref.subview %alloca_6[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_40 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %14 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_41 = memref.subview %alloca_10[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %15 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_42 = memref.subview %alloca_9[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %16 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_43 = memref.subview %alloca_8[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %17 = polygeist.submap(%arg2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_44 = memref.subview %alloca_6[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    linalg.generic {indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%14, %subview_41, %15, %subview_42, %16, %subview_43, %17 : memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_44 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %in_72: f64, %in_73: f64, %in_74: f64, %in_75: f64, %in_76: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.mulf %in_72, %in_73 : f64
      %31 = arith.addf %29, %30 : f64
      %32 = arith.mulf %in_74, %in_75 : f64
      %33 = arith.addf %31, %32 : f64
      %34 = arith.mulf %33, %in_76 : f64
      %35 = arith.addf %out, %34 : f64
      linalg.yield %35 : f64
    }
    %subview_45 = memref.subview %alloca_5[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_45 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %18 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_46 = memref.subview %alloca_10[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %19 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_47 = memref.subview %alloca_9[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %20 = polygeist.submap(%arg4, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_48 = memref.subview %alloca_8[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : memref<2x5x5x5xf64> to memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>
    %21 = polygeist.submap(%arg2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_49 = memref.subview %alloca_5[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    linalg.generic {indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%18, %subview_46, %19, %subview_47, %20, %subview_48, %21 : memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>, memref<?x?x?x?xf64, strided<[125, 25, 5, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_49 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %in_72: f64, %in_73: f64, %in_74: f64, %in_75: f64, %in_76: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.mulf %in_72, %in_73 : f64
      %31 = arith.addf %29, %30 : f64
      %32 = arith.mulf %in_74, %in_75 : f64
      %33 = arith.addf %31, %32 : f64
      %34 = arith.mulf %33, %in_76 : f64
      %35 = arith.addf %out, %34 : f64
      linalg.yield %35 : f64
    }
    %subview_50 = memref.subview %alloca_4[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_50 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_51 = memref.subview %alloca_7[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    %22 = polygeist.submap(%arg2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_52 = memref.subview %alloca_4[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_51, %22 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_52 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_53 = memref.subview %alloca_3[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_53 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_54 = memref.subview %alloca_6[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    %23 = polygeist.submap(%arg3, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_55 = memref.subview %alloca_3[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_54, %23 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_55 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_56 = memref.subview %alloca_2[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_56 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_57 = memref.subview %alloca_5[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : memref<2x5x5x4xf64> to memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>
    %24 = polygeist.submap(%arg2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_58 = memref.subview %alloca_2[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_57, %24 : memref<?x?x?x?xf64, strided<[100, 20, 4, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_58 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_59 = memref.subview %alloca_1[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_59 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_60 = memref.subview %alloca_4[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    %25 = polygeist.submap(%arg2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_61 = memref.subview %alloca_1[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_60, %25 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_61 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_62 = memref.subview %alloca_0[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_62 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_63 = memref.subview %alloca_3[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    %26 = polygeist.submap(%arg2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_64 = memref.subview %alloca_0[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_63, %26 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_64 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_65 = memref.subview %alloca[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_65 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_66 = memref.subview %alloca_2[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : memref<2x5x4x4xf64> to memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>
    %27 = polygeist.submap(%arg3, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?x?xf64>
    %subview_67 = memref.subview %alloca[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_66, %27 : memref<?x?x?x?xf64, strided<[80, 16, 4, 1]>>, memref<?x?x?x?x?xf64>) outs(%subview_67 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) {
    ^bb0(%in: f64, %in_71: f64, %out: f64):
      %29 = arith.mulf %in, %in_71 : f64
      %30 = arith.addf %out, %29 : f64
      linalg.yield %30 : f64
    }
    %subview_68 = memref.subview %alloca_1[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    %subview_69 = memref.subview %alloca_0[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    %subview_70 = memref.subview %alloca[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : memref<2x4x4x4xf64> to memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>
    %28 = polygeist.submap(%arg6, %c2, %c4, %c4, %c4) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_68, %subview_69, %subview_70 : memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>, memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>, memref<?x?x?x?xf64, strided<[64, 16, 4, 1]>>) outs(%28 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_71: f64, %in_72: f64, %out: f64):
      %29 = arith.addf %in, %in_71 : f64
      %30 = arith.addf %29, %in_72 : f64
      %31 = arith.addf %out, %30 : f64
      linalg.yield %31 : f64
    }
    return
  }
}

