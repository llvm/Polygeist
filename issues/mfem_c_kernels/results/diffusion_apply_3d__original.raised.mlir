#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + d1 * 5 + s0 * 750)>
#map8 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 125)>
#map9 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 250)>
#map10 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 375)>
#map11 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 500)>
#map12 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 625)>
#map13 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map14 = affine_map<(d0, d1, d2)[s0] -> (d0 * 5 + s0)>
#map15 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 64 + d0 * 16 + d1 * 4)>
#map16 = affine_map<(d0, d1, d2) -> (d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x3xf64>
    %alloca_0 = memref.alloca() : memref<4x4x3xf64>
    %alloca_1 = memref.alloca() : memref<5x2xf64>
    %alloca_2 = memref.alloca() : memref<5x5x3xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5x3xf64>
    affine.for %arg7 = 0 to 2 {
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_3 : memref<5x5x5x3xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg8 = 0 to 4 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_2 : memref<5x5x3xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 4 {
          %subview_6 = memref.subview %alloca_1[0, 1] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_6 : memref<?xf64, strided<[2], offset: 1>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %subview_7 = memref.subview %alloca_1[0, 0] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2]>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%subview_7 : memref<?xf64, strided<[2]>>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg10 = 0 to 4 {
            %8 = affine.load %arg5[%arg7 * 64 + %arg10 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
            %9 = polygeist.submap(%arg0, %arg10, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_15 = memref.subview %alloca_1[0, 0] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2]>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%9 : memref<?xf64>) outs(%subview_15 : memref<?xf64, strided<[2]>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %8, %in : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
            %10 = polygeist.submap(%arg1, %arg10, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_16 = memref.subview %alloca_1[0, 1] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2], offset: 1>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%10 : memref<?xf64>) outs(%subview_16 : memref<?xf64, strided<[2], offset: 1>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %8, %in : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
          }
          %subview_8 = memref.subview %alloca_1[0, 1] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          %subview_9 = memref.subview %alloca_1[0, 0] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2]>>
          %subview_10 = memref.subview %alloca_2[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x3xf64> to memref<?x?xf64, strided<[15, 3]>>
          %subview_11 = memref.subview %alloca_2[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: 1>>
          %subview_12 = memref.subview %alloca_2[0, 0, 2] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: 2>>
          %6 = polygeist.submap(%arg0, %arg9, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_13 = memref.subview %6[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          %7 = polygeist.submap(%arg1, %arg9, %c5) {map = #map3} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_14 = memref.subview %7[0] [%c5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
          linalg.generic {indexing_maps = [#map4, #map4, #map5, #map5, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%subview_13, %subview_14, %subview_8, %subview_9 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1]>>, memref<?xf64, strided<[2], offset: 1>>, memref<?xf64, strided<[2]>>) outs(%subview_10, %subview_11, %subview_12 : memref<?x?xf64, strided<[15, 3]>>, memref<?x?xf64, strided<[15, 3], offset: 1>>, memref<?x?xf64, strided<[15, 3], offset: 2>>) {
          ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64, %out_18: f64, %out_19: f64):
            %8 = arith.mulf %in_16, %in : f64
            %9 = arith.addf %out, %8 : f64
            %10 = arith.mulf %in_17, %in_15 : f64
            %11 = arith.addf %out_18, %10 : f64
            %12 = arith.mulf %in_17, %in : f64
            %13 = arith.addf %out_19, %12 : f64
            linalg.yield %9, %11, %13 : f64, f64, f64
          }
        }
        affine.for %arg9 = 0 to 5 {
          %6 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
          %7 = affine.load %arg1[%arg8 + %arg9 * 4] : memref<?xf64>
          %subview_6 = memref.subview %alloca_2[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x3xf64> to memref<?x?xf64, strided<[15, 3]>>
          %subview_7 = memref.subview %alloca_3[%arg9, 0, 0, 0] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%subview_6 : memref<?x?xf64, strided<[15, 3]>>) outs(%subview_7 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %6 : f64
            %9 = arith.addf %out, %8 : f64
            linalg.yield %9 : f64
          }
          %subview_8 = memref.subview %alloca_2[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: 1>>
          %subview_9 = memref.subview %alloca_3[%arg9, 0, 0, 1] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%subview_8 : memref<?x?xf64, strided<[15, 3], offset: 1>>) outs(%subview_9 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %6 : f64
            %9 = arith.addf %out, %8 : f64
            linalg.yield %9 : f64
          }
          %subview_10 = memref.subview %alloca_2[0, 0, 2] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: 2>>
          %subview_11 = memref.subview %alloca_3[%arg9, 0, 0, 2] [1, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?xf64, strided<[15, 3], offset: ?>>
          linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%subview_10 : memref<?x?xf64, strided<[15, 3], offset: 2>>) outs(%subview_11 : memref<?x?xf64, strided<[15, 3], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %8 = arith.mulf %in, %7 : f64
            %9 = arith.addf %out, %8 : f64
            linalg.yield %9 : f64
          }
        } {polygeist.was_parallel}
      }
      %0 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %1 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %2 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %3 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %4 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %5 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %subview = memref.subview %alloca_3[0, 0, 0, 0] [%c5, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?x?xf64, strided<[75, 15, 3]>>
      %subview_4 = memref.subview %alloca_3[0, 0, 0, 1] [%c5, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?x?xf64, strided<[75, 15, 3], offset: 1>>
      %subview_5 = memref.subview %alloca_3[0, 0, 0, 2] [%c5, %c5, %c5, 1] [1, 1, 1, 1] : memref<5x5x5x3xf64> to memref<?x?x?xf64, strided<[75, 15, 3], offset: 2>>
      linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1, %2, %3, %4, %5 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%subview, %subview_4, %subview_5 : memref<?x?x?xf64, strided<[75, 15, 3]>>, memref<?x?x?xf64, strided<[75, 15, 3], offset: 1>>, memref<?x?x?xf64, strided<[75, 15, 3], offset: 2>>) {
      ^bb0(%in: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %out: f64, %out_11: f64, %out_12: f64):
        %6 = arith.mulf %in, %out : f64
        %7 = arith.mulf %in_6, %out_11 : f64
        %8 = arith.addf %6, %7 : f64
        %9 = arith.mulf %in_7, %out_12 : f64
        %10 = arith.addf %8, %9 : f64
        %11 = arith.mulf %in_6, %out : f64
        %12 = arith.mulf %in_8, %out_11 : f64
        %13 = arith.addf %11, %12 : f64
        %14 = arith.mulf %in_9, %out_12 : f64
        %15 = arith.addf %13, %14 : f64
        %16 = arith.mulf %in_7, %out : f64
        %17 = arith.mulf %in_9, %out_11 : f64
        %18 = arith.addf %16, %17 : f64
        %19 = arith.mulf %in_10, %out_12 : f64
        %20 = arith.addf %18, %19 : f64
        linalg.yield %10, %15, %20 : f64, f64, f64
      }
      affine.for %arg8 = 0 to 5 {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_0 : memref<4x4x3xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 5 {
          linalg.generic {indexing_maps = [#map6], iterator_types = ["parallel", "parallel"]} outs(%alloca : memref<4x3xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg10 = 0 to 5 {
            %9 = affine.load %alloca_3[%arg8, %arg9, %arg10, 0] : memref<5x5x5x3xf64>
            %10 = affine.load %alloca_3[%arg8, %arg9, %arg10, 1] : memref<5x5x5x3xf64>
            %11 = affine.load %alloca_3[%arg8, %arg9, %arg10, 2] : memref<5x5x5x3xf64>
            %12 = polygeist.submap(%arg3, %arg10, %c4) {map = #map13} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_9 = memref.subview %alloca[0, 0] [%c4, 1] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[3]>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%12 : memref<?xf64>) outs(%subview_9 : memref<?xf64, strided<[3]>>) {
            ^bb0(%in: f64, %out: f64):
              %15 = arith.mulf %9, %in : f64
              %16 = arith.addf %out, %15 : f64
              linalg.yield %16 : f64
            }
            %13 = polygeist.submap(%arg2, %arg10, %c4) {map = #map13} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_10 = memref.subview %alloca[0, 1] [%c4, 1] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[3], offset: 1>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%13 : memref<?xf64>) outs(%subview_10 : memref<?xf64, strided<[3], offset: 1>>) {
            ^bb0(%in: f64, %out: f64):
              %15 = arith.mulf %10, %in : f64
              %16 = arith.addf %out, %15 : f64
              linalg.yield %16 : f64
            }
            %14 = polygeist.submap(%arg2, %arg10, %c4) {map = #map13} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_11 = memref.subview %alloca[0, 2] [%c4, 1] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[3], offset: 2>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%14 : memref<?xf64>) outs(%subview_11 : memref<?xf64, strided<[3], offset: 2>>) {
            ^bb0(%in: f64, %out: f64):
              %15 = arith.mulf %11, %in : f64
              %16 = arith.addf %out, %15 : f64
              linalg.yield %16 : f64
            }
          }
          affine.for %arg10 = 0 to 4 {
            %9 = affine.load %arg2[%arg9 + %arg10 * 5] : memref<?xf64>
            %10 = affine.load %arg3[%arg9 + %arg10 * 5] : memref<?xf64>
            %subview_9 = memref.subview %alloca[0, 0] [%c4, 1] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[3]>>
            %subview_10 = memref.subview %alloca_0[%arg10, 0, 0] [1, %c4, 1] [1, 1, 1] : memref<4x4x3xf64> to memref<?xf64, strided<[3], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_9 : memref<?xf64, strided<[3]>>) outs(%subview_10 : memref<?xf64, strided<[3], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %9 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
            %subview_11 = memref.subview %alloca[0, 1] [%c4, 1] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[3], offset: 1>>
            %subview_12 = memref.subview %alloca_0[%arg10, 0, 1] [1, %c4, 1] [1, 1, 1] : memref<4x4x3xf64> to memref<?xf64, strided<[3], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_11 : memref<?xf64, strided<[3], offset: 1>>) outs(%subview_12 : memref<?xf64, strided<[3], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %10 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
            %subview_13 = memref.subview %alloca[0, 2] [%c4, 1] [1, 1] : memref<4x3xf64> to memref<?xf64, strided<[3], offset: 2>>
            %subview_14 = memref.subview %alloca_0[%arg10, 0, 2] [1, %c4, 1] [1, 1, 1] : memref<4x4x3xf64> to memref<?xf64, strided<[3], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview_13 : memref<?xf64, strided<[3], offset: 2>>) outs(%subview_14 : memref<?xf64, strided<[3], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %11 = arith.mulf %in, %9 : f64
              %12 = arith.addf %out, %11 : f64
              linalg.yield %12 : f64
            }
          } {polygeist.was_parallel}
        }
        %subview_6 = memref.subview %alloca_0[0, 0, 0] [%c4, %c4, 1] [1, 1, 1] : memref<4x4x3xf64> to memref<?x?xf64, strided<[12, 3]>>
        %subview_7 = memref.subview %alloca_0[0, 0, 1] [%c4, %c4, 1] [1, 1, 1] : memref<4x4x3xf64> to memref<?x?xf64, strided<[12, 3], offset: 1>>
        %6 = polygeist.submap(%arg2, %arg8, %c4, %c4, %c4) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %subview_8 = memref.subview %alloca_0[0, 0, 2] [%c4, %c4, 1] [1, 1, 1] : memref<4x4x3xf64> to memref<?x?xf64, strided<[12, 3], offset: 2>>
        %7 = polygeist.submap(%arg3, %arg8, %c4, %c4, %c4) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %8 = polygeist.submap(%arg6, %arg7, %c4, %c4, %c4) {map = #map15} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map16, #map16, #map1, #map16, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_6, %subview_7, %6, %subview_8, %7 : memref<?x?xf64, strided<[12, 3]>>, memref<?x?xf64, strided<[12, 3], offset: 1>>, memref<?x?x?xf64>, memref<?x?xf64, strided<[12, 3], offset: 2>>, memref<?x?x?xf64>) outs(%8 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %out: f64):
          %9 = arith.addf %in, %in_9 : f64
          %10 = arith.mulf %9, %in_10 : f64
          %11 = arith.mulf %in_11, %in_12 : f64
          %12 = arith.addf %10, %11 : f64
          %13 = arith.addf %out, %12 : f64
          linalg.yield %13 : f64
        }
      }
    }
    return
  }
}
