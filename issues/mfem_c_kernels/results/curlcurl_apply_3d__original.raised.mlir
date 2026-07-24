#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1 * 3 + d0)>
#map4 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 12 + s1 * 3 + s2 * 144)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map8 = affine_map<(d0)[s0, s1, s2] -> (d0 * 4 + s0 * 12 + s1 + s2 * 144 + 48)>
#map9 = affine_map<(d0, d1) -> (d1, d0)>
#map10 = affine_map<(d0)[s0, s1, s2] -> (d0 * 16 + s0 + s1 * 4 + s2 * 144 + 96)>
#map11 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5)>
#map12 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 125)>
#map13 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 250)>
#map14 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 375)>
#map15 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 500)>
#map16 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 625)>
#map17 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map18 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map19 = affine_map<(d0, d1, d2)[s0] -> (d0 * 5 + s0)>
#map20 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 12 + d1 * 3 + s0 * 144)>
#map21 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map22 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 12 + d1 * 4 + s0 * 144 + 48)>
#map23 = affine_map<(d0, d1, d2)[s0] -> (d2 * 16 + d0 + d1 * 4 + s0 * 144 + 96)>
#map24 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_curlcurl_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<3x2xf64>
    %alloca_0 = memref.alloca() : memref<3x4xf64>
    %alloca_1 = memref.alloca() : memref<3x4xf64>
    %alloca_2 = memref.alloca() : memref<3x2xf64>
    %alloca_3 = memref.alloca() : memref<3x4xf64>
    %alloca_4 = memref.alloca() : memref<3x4xf64>
    %alloca_5 = memref.alloca() : memref<3x2xf64>
    %alloca_6 = memref.alloca() : memref<4x3xf64>
    %alloca_7 = memref.alloca() : memref<4x3xf64>
    %alloca_8 = memref.alloca() : memref<5xf64>
    %alloca_9 = memref.alloca() : memref<5x5x2xf64>
    %alloca_10 = memref.alloca() : memref<5xf64>
    %alloca_11 = memref.alloca() : memref<5x5x2xf64>
    %alloca_12 = memref.alloca() : memref<5xf64>
    %alloca_13 = memref.alloca() : memref<5x5x2xf64>
    %alloca_14 = memref.alloca() : memref<5x5x5x3xf64>
    affine.for %arg9 = 0 to 2 {
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_14 : memref<5x5x5x3xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg10 = 0 to 4 {
        %subview_17 = memref.subview %alloca_13[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_17 : memref<?x?xf64, strided<[10, 2], offset: 1>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %subview_18 = memref.subview %alloca_13[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_18 : memref<?x?xf64, strided<[10, 2]>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 4 {
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca_12 : memref<5xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %6 = polygeist.submap(%arg0, %c3, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%arg7, %arg10, %arg11, %arg9, %c3) {map = #map4} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map5, #map1, #map6], iterator_types = ["reduction", "parallel"]} ins(%7, %6 : memref<?xf64>, memref<?x?xf64>) outs(%alloca_12 : memref<5xf64>) {
          ^bb0(%in: f64, %in_24: f64, %out: f64):
            %10 = arith.mulf %in, %in_24 : f64
            %11 = arith.addf %out, %10 : f64
            linalg.yield %11 : f64
          }
          %subview_19 = memref.subview %alloca_12[0] [%c5] [1] : memref<5xf64> to memref<?xf64, strided<[1]>>
          %subview_20 = memref.subview %alloca_13[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
          %subview_21 = memref.subview %alloca_13[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
          %8 = polygeist.submap(%arg4, %arg11, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_22 = memref.subview %8[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          %9 = polygeist.submap(%arg1, %arg11, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_23 = memref.subview %9[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          linalg.generic {indexing_maps = [#map5, #map5, #map6, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_22, %subview_23, %subview_19 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>) outs(%subview_20, %subview_21 : memref<?x?xf64, strided<[10, 2]>>, memref<?x?xf64, strided<[10, 2], offset: 1>>) {
          ^bb0(%in: f64, %in_24: f64, %in_25: f64, %out: f64, %out_26: f64):
            %10 = arith.mulf %in_25, %in : f64
            %11 = arith.addf %out, %10 : f64
            %12 = arith.mulf %in_25, %in_24 : f64
            %13 = arith.addf %out_26, %12 : f64
            linalg.yield %11, %13 : f64, f64
          }
        }
        affine.for %arg11 = 0 to 5 {
          %6 = affine.load %arg4[%arg10 + %arg11 * 4] : memref<?xf64>
          %7 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
          %subview_19 = memref.subview %alloca_13[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
          %subview_20 = memref.subview %alloca_14[%arg11, 0, 0, 1] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_19 : memref<?x?xf64, strided<[10, 2], offset: 1>>) outs(%subview_20 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %6 : f64
            %9 = arith.addf %out, %8 : f64
            linalg.yield %9 : f64
          }
          %subview_21 = memref.subview %alloca_13[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
          %subview_22 = memref.subview %alloca_14[%arg11, 0, 0, 2] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_21 : memref<?x?xf64, strided<[10, 2]>>) outs(%subview_22 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %7 : f64
            %9 = arith.subf %out, %8 : f64
            linalg.yield %9 : f64
          }
        } {polygeist.was_parallel}
      }
      affine.for %arg10 = 0 to 4 {
        %subview_17 = memref.subview %alloca_11[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_17 : memref<?x?xf64, strided<[10, 2], offset: 1>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %subview_18 = memref.subview %alloca_11[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_18 : memref<?x?xf64, strided<[10, 2]>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 4 {
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca_10 : memref<5xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %6 = polygeist.submap(%arg0, %c3, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%arg7, %arg10, %arg11, %arg9, %c3) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map5, #map1, #map6], iterator_types = ["reduction", "parallel"]} ins(%7, %6 : memref<?xf64>, memref<?x?xf64>) outs(%alloca_10 : memref<5xf64>) {
          ^bb0(%in: f64, %in_24: f64, %out: f64):
            %10 = arith.mulf %in, %in_24 : f64
            %11 = arith.addf %out, %10 : f64
            linalg.yield %11 : f64
          }
          %subview_19 = memref.subview %alloca_10[0] [%c5] [1] : memref<5xf64> to memref<?xf64, strided<[1]>>
          %subview_20 = memref.subview %alloca_11[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
          %subview_21 = memref.subview %alloca_11[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
          %8 = polygeist.submap(%arg4, %arg11, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_22 = memref.subview %8[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          %9 = polygeist.submap(%arg1, %arg11, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_23 = memref.subview %9[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          linalg.generic {indexing_maps = [#map5, #map5, #map6, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%subview_22, %subview_23, %subview_19 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>) outs(%subview_20, %subview_21 : memref<?x?xf64, strided<[10, 2]>>, memref<?x?xf64, strided<[10, 2], offset: 1>>) {
          ^bb0(%in: f64, %in_24: f64, %in_25: f64, %out: f64, %out_26: f64):
            %10 = arith.mulf %in, %in_25 : f64
            %11 = arith.addf %out, %10 : f64
            %12 = arith.mulf %in_24, %in_25 : f64
            %13 = arith.addf %out_26, %12 : f64
            linalg.yield %11, %13 : f64, f64
          }
        }
        affine.for %arg11 = 0 to 5 {
          %6 = affine.load %arg4[%arg10 + %arg11 * 4] : memref<?xf64>
          %7 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
          %subview_19 = memref.subview %alloca_11[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
          %subview_20 = memref.subview %alloca_14[%arg11, 0, 0, 0] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_19 : memref<?x?xf64, strided<[10, 2], offset: 1>>) outs(%subview_20 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %6 : f64
            %9 = arith.subf %out, %8 : f64
            linalg.yield %9 : f64
          }
          %subview_21 = memref.subview %alloca_11[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
          %subview_22 = memref.subview %alloca_14[%arg11, 0, 0, 2] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_21 : memref<?x?xf64, strided<[10, 2]>>) outs(%subview_22 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %7 : f64
            %9 = arith.addf %out, %8 : f64
            linalg.yield %9 : f64
          }
        } {polygeist.was_parallel}
      }
      affine.for %arg10 = 0 to 4 {
        %subview_17 = memref.subview %alloca_9[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_17 : memref<?x?xf64, strided<[10, 2], offset: 1>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %subview_18 = memref.subview %alloca_9[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_18 : memref<?x?xf64, strided<[10, 2]>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 4 {
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca_8 : memref<5xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %6 = polygeist.submap(%arg0, %c3, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%arg7, %arg10, %arg11, %arg9, %c3) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map5, #map1, #map6], iterator_types = ["reduction", "parallel"]} ins(%7, %6 : memref<?xf64>, memref<?x?xf64>) outs(%alloca_8 : memref<5xf64>) {
          ^bb0(%in: f64, %in_24: f64, %out: f64):
            %10 = arith.mulf %in, %in_24 : f64
            %11 = arith.addf %out, %10 : f64
            linalg.yield %11 : f64
          }
          %subview_19 = memref.subview %alloca_8[0] [%c5] [1] : memref<5xf64> to memref<?xf64, strided<[1]>>
          %subview_20 = memref.subview %alloca_9[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
          %subview_21 = memref.subview %alloca_9[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
          %8 = polygeist.submap(%arg1, %arg11, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_22 = memref.subview %8[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          %9 = polygeist.submap(%arg4, %arg11, %c5) {map = #map7} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_23 = memref.subview %9[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          linalg.generic {indexing_maps = [#map5, #map5, #map6, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%subview_22, %subview_23, %subview_19 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>) outs(%subview_20, %subview_21 : memref<?x?xf64, strided<[10, 2]>>, memref<?x?xf64, strided<[10, 2], offset: 1>>) {
          ^bb0(%in: f64, %in_24: f64, %in_25: f64, %out: f64, %out_26: f64):
            %10 = arith.mulf %in_25, %in : f64
            %11 = arith.addf %out, %10 : f64
            %12 = arith.mulf %in_25, %in_24 : f64
            %13 = arith.addf %out_26, %12 : f64
            linalg.yield %11, %13 : f64, f64
          }
        }
        affine.for %arg11 = 0 to 5 {
          %6 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
          %7 = affine.load %arg4[%arg10 + %arg11 * 4] : memref<?xf64>
          %subview_19 = memref.subview %alloca_9[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
          %subview_20 = memref.subview %alloca_14[0, 0, %arg11, 0] [%c5, %c5, 1, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[75, 15], offset: ?>>
          linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%subview_19 : memref<?x?xf64, strided<[10, 2], offset: 1>>) outs(%subview_20 : memref<?x?xf64, strided<[75, 15], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %6 : f64
            %9 = arith.addf %out, %8 : f64
            linalg.yield %9 : f64
          }
          %subview_21 = memref.subview %alloca_9[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
          %subview_22 = memref.subview %alloca_14[0, 0, %arg11, 1] [%c5, %c5, 1, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[75, 15], offset: ?>>
          linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%subview_21 : memref<?x?xf64, strided<[10, 2]>>) outs(%subview_22 : memref<?x?xf64, strided<[75, 15], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %7 : f64
            %9 = arith.subf %out, %8 : f64
            linalg.yield %9 : f64
          }
        } {polygeist.was_parallel}
      }
      %0 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %1 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %2 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map13} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %3 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %4 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map15} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %5 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %subview = memref.subview %alloca_14[0, 0, 0, 0] [%c5, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?x?xf64, strided<[75, 15, 3]>>
      %subview_15 = memref.subview %alloca_14[0, 0, 0, 1] [%c5, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?x?xf64, strided<[75, 15, 3], offset: 1>>
      %subview_16 = memref.subview %alloca_14[0, 0, 0, 2] [%c5, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?x?xf64, strided<[75, 15, 3], offset: 2>>
      linalg.generic {indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1, %2, %3, %4, %5 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%subview, %subview_15, %subview_16 : memref<?x?x?xf64, strided<[75, 15, 3]>>, memref<?x?x?xf64, strided<[75, 15, 3], offset: 1>>, memref<?x?x?xf64, strided<[75, 15, 3], offset: 2>>) {
      ^bb0(%in: f64, %in_17: f64, %in_18: f64, %in_19: f64, %in_20: f64, %in_21: f64, %out: f64, %out_22: f64, %out_23: f64):
        %6 = arith.mulf %in, %out : f64
        %7 = arith.mulf %in_17, %out_22 : f64
        %8 = arith.addf %6, %7 : f64
        %9 = arith.mulf %in_18, %out_23 : f64
        %10 = arith.addf %8, %9 : f64
        %11 = arith.mulf %in_17, %out : f64
        %12 = arith.mulf %in_19, %out_22 : f64
        %13 = arith.addf %11, %12 : f64
        %14 = arith.mulf %in_20, %out_23 : f64
        %15 = arith.addf %13, %14 : f64
        %16 = arith.mulf %in_18, %out : f64
        %17 = arith.mulf %in_20, %out_22 : f64
        %18 = arith.addf %16, %17 : f64
        %19 = arith.mulf %in_21, %out_23 : f64
        %20 = arith.addf %18, %19 : f64
        linalg.yield %10, %15, %20 : f64, f64, f64
      }
      affine.for %arg10 = 0 to 5 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_6 : memref<4x3xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_7 : memref<4x3xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 5 {
          %subview_17 = memref.subview %alloca_5[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_17 : memref<?xf64, strided<[2], offset: 1>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %subview_18 = memref.subview %alloca_5[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_18 : memref<?xf64, strided<[2]>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg12 = 0 to 5 {
            %9 = affine.load %alloca_14[%arg10, %arg11, %arg12, 1] : memref<5x5x5x3xf64>
            %10 = affine.load %alloca_14[%arg10, %arg11, %arg12, 2] : memref<5x5x5x3xf64>
            %11 = polygeist.submap(%arg2, %arg12, %c3) {map = #map18} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_19 = memref.subview %alloca_5[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%11 : memref<?xf64>) outs(%subview_19 : memref<?xf64, strided<[2]>>) {
            ^bb0(%in: f64, %out: f64):
              %13 = arith.mulf %in, %9 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14 : f64
            }
            %12 = polygeist.submap(%arg2, %arg12, %c3) {map = #map18} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_20 = memref.subview %alloca_5[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%12 : memref<?xf64>) outs(%subview_20 : memref<?xf64, strided<[2], offset: 1>>) {
            ^bb0(%in: f64, %out: f64):
              %13 = arith.mulf %in, %10 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14 : f64
            }
          }
          affine.for %arg12 = 0 to 4 {
            %9 = affine.load %arg3[%arg11 + %arg12 * 5] : memref<?xf64>
            %10 = affine.load %arg5[%arg11 + %arg12 * 5] : memref<?xf64>
            %subview_19 = memref.subview %alloca_5[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
            %subview_20 = memref.subview %alloca_6[%arg12, 0] [1, %c3] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_19 : memref<?xf64, strided<[2]>>) outs(%subview_20 : memref<?xf64, strided<[1], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %9 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
            %subview_21 = memref.subview %alloca_5[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            %subview_22 = memref.subview %alloca_7[%arg12, 0] [1, %c3] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_21 : memref<?xf64, strided<[2], offset: 1>>) outs(%subview_22 : memref<?xf64, strided<[1], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %10 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
          } {polygeist.was_parallel}
        }
        %6 = polygeist.submap(%arg5, %arg10, %c4, %c4, %c3) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %7 = polygeist.submap(%arg3, %arg10, %c4, %c4, %c3) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %8 = polygeist.submap(%arg8, %arg9, %c4, %c4, %c3) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map21, #map17, #map21, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloca_6, %6, %alloca_7, %7 : memref<4x3xf64>, memref<?x?x?xf64>, memref<4x3xf64>, memref<?x?x?xf64>) outs(%8 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_17: f64, %in_18: f64, %in_19: f64, %out: f64):
          %9 = arith.mulf %in, %in_17 : f64
          %10 = arith.mulf %in_18, %in_19 : f64
          %11 = arith.subf %9, %10 : f64
          %12 = arith.addf %out, %11 : f64
          linalg.yield %12 : f64
        }
      }
      affine.for %arg10 = 0 to 5 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_3 : memref<3x4xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_4 : memref<3x4xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 5 {
          %subview_17 = memref.subview %alloca_2[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_17 : memref<?xf64, strided<[2], offset: 1>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %subview_18 = memref.subview %alloca_2[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_18 : memref<?xf64, strided<[2]>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg12 = 0 to 5 {
            %9 = affine.load %alloca_14[%arg10, %arg12, %arg11, 2] : memref<5x5x5x3xf64>
            %10 = affine.load %alloca_14[%arg10, %arg12, %arg11, 0] : memref<5x5x5x3xf64>
            %11 = polygeist.submap(%arg2, %arg12, %c3) {map = #map18} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_19 = memref.subview %alloca_2[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%11 : memref<?xf64>) outs(%subview_19 : memref<?xf64, strided<[2]>>) {
            ^bb0(%in: f64, %out: f64):
              %13 = arith.mulf %in, %9 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14 : f64
            }
            %12 = polygeist.submap(%arg2, %arg12, %c3) {map = #map18} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_20 = memref.subview %alloca_2[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%12 : memref<?xf64>) outs(%subview_20 : memref<?xf64, strided<[2], offset: 1>>) {
            ^bb0(%in: f64, %out: f64):
              %13 = arith.mulf %in, %10 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14 : f64
            }
          }
          affine.for %arg12 = 0 to 4 {
            %9 = affine.load %arg5[%arg11 + %arg12 * 5] : memref<?xf64>
            %10 = affine.load %arg3[%arg11 + %arg12 * 5] : memref<?xf64>
            %subview_19 = memref.subview %alloca_2[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
            %subview_20 = memref.subview %alloca_4[0, %arg12] [%c3, 1] [1, 1] : memref<3x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_19 : memref<?xf64, strided<[2]>>) outs(%subview_20 : memref<?xf64, strided<[4], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %9 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
            %subview_21 = memref.subview %alloca_2[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            %subview_22 = memref.subview %alloca_3[0, %arg12] [%c3, 1] [1, 1] : memref<3x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_21 : memref<?xf64, strided<[2], offset: 1>>) outs(%subview_22 : memref<?xf64, strided<[4], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %10 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
          } {polygeist.was_parallel}
        }
        %6 = polygeist.submap(%arg5, %arg10, %c4, %c3, %c4) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %7 = polygeist.submap(%arg3, %arg10, %c4, %c3, %c4) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %8 = polygeist.submap(%arg8, %arg9, %c4, %c3, %c4) {map = #map22} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map21, #map17, #map21, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloca_3, %6, %alloca_4, %7 : memref<3x4xf64>, memref<?x?x?xf64>, memref<3x4xf64>, memref<?x?x?xf64>) outs(%8 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_17: f64, %in_18: f64, %in_19: f64, %out: f64):
          %9 = arith.negf %in : f64
          %10 = arith.mulf %9, %in_17 : f64
          %11 = arith.mulf %in_18, %in_19 : f64
          %12 = arith.addf %10, %11 : f64
          %13 = arith.addf %out, %12 : f64
          linalg.yield %13 : f64
        }
      }
      affine.for %arg10 = 0 to 5 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_0 : memref<3x4xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_1 : memref<3x4xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 5 {
          %subview_17 = memref.subview %alloca[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_17 : memref<?xf64, strided<[2], offset: 1>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %subview_18 = memref.subview %alloca[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_18 : memref<?xf64, strided<[2]>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg12 = 0 to 5 {
            %9 = affine.load %alloca_14[%arg12, %arg11, %arg10, 0] : memref<5x5x5x3xf64>
            %10 = affine.load %alloca_14[%arg12, %arg11, %arg10, 1] : memref<5x5x5x3xf64>
            %11 = polygeist.submap(%arg2, %arg12, %c3) {map = #map18} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_19 = memref.subview %alloca[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%11 : memref<?xf64>) outs(%subview_19 : memref<?xf64, strided<[2]>>) {
            ^bb0(%in: f64, %out: f64):
              %13 = arith.mulf %in, %9 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14 : f64
            }
            %12 = polygeist.submap(%arg2, %arg12, %c3) {map = #map18} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_20 = memref.subview %alloca[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%12 : memref<?xf64>) outs(%subview_20 : memref<?xf64, strided<[2], offset: 1>>) {
            ^bb0(%in: f64, %out: f64):
              %13 = arith.mulf %in, %10 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14 : f64
            }
          }
          affine.for %arg12 = 0 to 4 {
            %9 = affine.load %arg3[%arg11 + %arg12 * 5] : memref<?xf64>
            %10 = affine.load %arg5[%arg11 + %arg12 * 5] : memref<?xf64>
            %subview_19 = memref.subview %alloca[0, 1] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            %subview_20 = memref.subview %alloca_1[0, %arg12] [%c3, 1] [1, 1] : memref<3x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_19 : memref<?xf64, strided<[2], offset: 1>>) outs(%subview_20 : memref<?xf64, strided<[4], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %9, %in : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
            %subview_21 = memref.subview %alloca[0, 0] [%c3, 1] [1, 1] : memref<3x2xf64> to memref<?xf64, strided<[2]>>
            %subview_22 = memref.subview %alloca_0[0, %arg12] [%c3, 1] [1, 1] : memref<3x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_21 : memref<?xf64, strided<[2]>>) outs(%subview_22 : memref<?xf64, strided<[4], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %10, %in : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
          } {polygeist.was_parallel}
        }
        %6 = polygeist.submap(%arg3, %arg10, %c4, %c4, %c3) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %7 = polygeist.submap(%arg5, %arg10, %c4, %c4, %c3) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %8 = polygeist.submap(%arg8, %arg9, %c4, %c4, %c3) {map = #map23} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map24, #map17, #map24, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloca_0, %6, %alloca_1, %7 : memref<3x4xf64>, memref<?x?x?xf64>, memref<3x4xf64>, memref<?x?x?xf64>) outs(%8 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_17: f64, %in_18: f64, %in_19: f64, %out: f64):
          %9 = arith.mulf %in, %in_17 : f64
          %10 = arith.mulf %in_18, %in_19 : f64
          %11 = arith.subf %9, %10 : f64
          %12 = arith.addf %out, %11 : f64
          linalg.yield %12 : f64
        }
      }
    }
    return
  }
}
