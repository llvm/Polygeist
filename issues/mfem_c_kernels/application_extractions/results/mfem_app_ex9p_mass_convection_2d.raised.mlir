#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2 + d0 * 25 + d1 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map11 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5 + 25)>
#map14 = affine_map<(d0) -> (d0)>
#map15 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex9p_mass_convection_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: f64, %arg8: f64, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>, %arg12: memref<?xf64>, %arg13: memref<?xf64>, %arg14: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x5xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5xf64>
    %subview = memref.subview %alloca_1[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %1 = polygeist.submap(%arg5, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_1 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %subview_2 = memref.subview %alloca_0[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_2 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_3 = memref.subview %alloca_1[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %subview_4 = memref.subview %alloca_0[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2, %subview_3 : memref<?x?x?x?xf64>, memref<?x?x?xf64, strided<[20, 5, 1]>>) outs(%subview_4 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %3 = polygeist.submap(%arg3, %c2, %c5, %c5) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    %subview_5 = memref.subview %alloca_0[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : memref<?x?x?xf64>) outs(%subview_5 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%in: f64, %out: f64):
      %18 = arith.mulf %out, %in : f64
      linalg.yield %18 : f64
    }
    %subview_6 = memref.subview %alloca[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_6 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %4 = polygeist.submap(%arg2, %c2, %c5, %c4, %c5) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_7 = memref.subview %alloca_0[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    %subview_8 = memref.subview %alloca[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map3, #map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4, %subview_7 : memref<?x?x?x?xf64>, memref<?x?x?xf64, strided<[25, 5, 1]>>) outs(%subview_8 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %5 = polygeist.submap(%arg2, %c2, %c4, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_9 = memref.subview %alloca[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %6 = polygeist.submap(%arg9, %c2, %c4, %c4) {map = #map11} : (memref<?xf64>, index, index, index) -> memref<2x4x4xf64>
    linalg.generic {indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %subview_9 : memref<?x?x?x?xf64>, memref<?x?x?xf64, strided<[20, 4, 1]>>) outs(%6 : memref<2x4x4xf64>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %alloca_10 = memref.alloca() : memref<2x4x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x5xf64>
    %alloca_13 = memref.alloca() : memref<2x5x5xf64>
    %alloca_14 = memref.alloca() : memref<2x4x5xf64>
    %alloca_15 = memref.alloca() : memref<2x4x5xf64>
    %subview_16 = memref.subview %alloca_15[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_16 : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %7 = polygeist.submap(%arg5, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %8 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%7, %8 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_15 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %subview_17 = memref.subview %alloca_14[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_17 : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %9 = polygeist.submap(%arg5, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %10 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_14 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %subview_18 = memref.subview %alloca_13[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_18 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_19 = memref.subview %alloca_14[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %11 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_20 = memref.subview %alloca_13[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_19, %11 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%subview_20 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %subview_21 = memref.subview %alloca_12[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_21 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_22 = memref.subview %alloca_15[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %12 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_23 = memref.subview %alloca_12[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_22, %12 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%subview_23 : memref<?x?x?xf64, strided<[25, 5, 1]>>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %subview_24 = memref.subview %alloca_11[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_24 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %13 = polygeist.submap(%arg4, %c2, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_25 = memref.subview %alloca_13[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    %14 = polygeist.submap(%arg4, %c2, %c5, %c4, %c5) {map = #map13} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_26 = memref.subview %alloca_12[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : memref<2x5x5xf64> to memref<?x?x?xf64, strided<[25, 5, 1]>>
    %15 = polygeist.submap(%arg2, %c2, %c5, %c4, %c5) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_27 = memref.subview %alloca_11[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map3, #map9, #map3, #map9, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%13, %subview_25, %14, %subview_26, %15 : memref<?x?x?x?xf64>, memref<?x?x?xf64, strided<[25, 5, 1]>>, memref<?x?x?x?xf64>, memref<?x?x?xf64, strided<[25, 5, 1]>>, memref<?x?x?x?xf64>) outs(%subview_27 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%in: f64, %in_40: f64, %in_41: f64, %in_42: f64, %in_43: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.mulf %in_41, %in_42 : f64
      %20 = arith.addf %18, %19 : f64
      %21 = arith.mulf %20, %in_43 : f64
      %22 = arith.addf %out, %21 : f64
      linalg.yield %22 : f64
    }
    %subview_28 = memref.subview %alloca_10[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_28 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_29 = memref.subview %alloca_11[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %16 = polygeist.submap(%arg2, %c2, %c4, %c4, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_30 = memref.subview %alloca_10[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_29, %16 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_30 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_40: f64, %out: f64):
      %18 = arith.mulf %in, %in_40 : f64
      %19 = arith.addf %out, %18 : f64
      linalg.yield %19 : f64
    }
    %subview_31 = memref.subview %alloca_10[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %17 = polygeist.submap(%arg10, %c2, %c4, %c4) {map = #map11} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_31 : memref<?x?x?xf64, strided<[16, 4, 1]>>) outs(%17 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %18 = arith.addf %out, %in : f64
      linalg.yield %18 : f64
    }
    affine.store %cst, %arg14[0] : memref<?xf64>
    %subview_32 = memref.subview %arg13[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_33 = memref.subview %arg9[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_34 = memref.subview %arg6[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_35 = memref.subview %arg11[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_36 = memref.subview %arg10[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_37 = memref.subview %arg11[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_38 = memref.subview %arg12[0] [%c32] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_39 = memref.subview %arg14[0] [1] [1] : memref<?xf64> to memref<f64, strided<[]>>
    linalg.generic {indexing_maps = [#map14, #map14, #map14, #map14, #map14, #map14, #map14, #map15], iterator_types = ["reduction"]} ins(%subview_32, %subview_33, %subview_34, %subview_35 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>) outs(%subview_36, %subview_37, %subview_38, %subview_39 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<f64, strided<[]>>) {
    ^bb0(%in: f64, %in_40: f64, %in_41: f64, %in_42: f64, %out: f64, %out_43: f64, %out_44: f64, %out_45: f64):
      %18 = arith.mulf %arg7, %in : f64
      %19 = arith.addf %out, %18 : f64
      %20 = arith.mulf %arg7, %in_40 : f64
      %21 = arith.subf %out_43, %20 : f64
      %22 = arith.mulf %in_41, %21 : f64
      %23 = arith.mulf %in_42, %22 : f64
      %24 = arith.addf %out_45, %23 : f64
      linalg.yield %19, %21, %22, %24 : f64, f64, f64, f64
    }
    linalg.generic {indexing_maps = [#map14, #map14], iterator_types = ["parallel"]} ins(%arg12 : memref<?xf64>) outs(%arg13 : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %18 = arith.mulf %arg8, %out : f64
      %19 = arith.addf %in, %18 : f64
      linalg.yield %19 : f64
    }
    return
  }
}

