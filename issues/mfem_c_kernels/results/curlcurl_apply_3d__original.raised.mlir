#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1, 1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1, 0)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d1 * 3 + d0)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 12 + s1 * 3 + s2 * 144)>
#map8 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d2, 1)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 1)>
#map12 = affine_map<(d0, d1, d2) -> (d0)>
#map13 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map14 = affine_map<(d0, d1, d2) -> (d1, d2, 0)>
#map15 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 2)>
#map16 = affine_map<(d0)[s0, s1, s2] -> (d0 * 4 + s0 * 12 + s1 + s2 * 144 + 48)>
#map17 = affine_map<(d0, d1) -> (d1, d0, 0)>
#map18 = affine_map<(d0, d1) -> (d1, d0, 1)>
#map19 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 0)>
#map20 = affine_map<(d0)[s0, s1, s2] -> (d0 * 16 + s0 + s1 * 4 + s2 * 144 + 96)>
#map21 = affine_map<(d0, d1, d2) -> (d2, d1, 1)>
#map22 = affine_map<(d0, d1, d2) -> (d2, d1, d0, 0)>
#map23 = affine_map<(d0, d1, d2) -> (d2, d1, 0)>
#map24 = affine_map<(d0, d1, d2) -> (d2, d1, d0, 1)>
#map25 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5)>
#map26 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 125)>
#map27 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 250)>
#map28 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 375)>
#map29 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 500)>
#map30 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 750 + d0 * 25 + d1 * 5 + 625)>
#map31 = affine_map<(d0) -> (d0, 1)>
#map32 = affine_map<(d0) -> (d0, 0)>
#map33 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map34 = affine_map<(d0, d1) -> (d1, 0)>
#map35 = affine_map<(d0, d1) -> (d1, 1)>
#map36 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map37 = affine_map<(d0, d1, d2)[s0] -> (d0 * 5 + s0)>
#map38 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 12 + d1 * 3 + s0 * 144)>
#map39 = affine_map<(d0, d1) -> (d1, d0)>
#map40 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 12 + d1 * 4 + s0 * 144 + 48)>
#map41 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map42 = affine_map<(d0, d1, d2)[s0] -> (d2 * 16 + d0 + d1 * 4 + s0 * 144 + 96)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      %0 = polygeist.submap(%alloca_14, %c5, %c5, %c5, %c3) {map = #map} : (memref<5x5x5x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg10 = 0 to 4 {
        %10 = polygeist.submap(%alloca_13, %c5, %c5) {map = #map1} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%10 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %11 = polygeist.submap(%alloca_13, %c5, %c5) {map = #map3} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%11 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 4 {
          %20 = polygeist.submap(%alloca_12, %c5) {map = #map4} : (memref<5xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%20 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %21 = polygeist.submap(%arg0, %c3, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %22 = polygeist.submap(%alloca_12, %c3, %c5) {map = #map6} : (memref<5xf64>, index, index) -> memref<?x?xf64>
          %23 = polygeist.submap(%arg7, %arg10, %arg11, %arg9, %c3) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          %24 = polygeist.submap(%23, %c3, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction", "parallel"]} ins(%24, %21 : memref<?x?xf64>, memref<?x?xf64>) outs(%22 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %32 = arith.mulf %in, %in_15 : f64
            %33 = arith.addf %out, %32 : f64
            linalg.yield %33 : f64
          }
          %25 = polygeist.submap(%alloca_12, %c5, %c5) {map = #map6} : (memref<5xf64>, index, index) -> memref<?x?xf64>
          %26 = polygeist.submap(%alloca_13, %c5, %c5) {map = #map3} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
          %27 = polygeist.submap(%alloca_13, %c5, %c5) {map = #map1} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
          %28 = polygeist.submap(%arg4, %arg11, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
          %29 = polygeist.submap(%28, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %30 = polygeist.submap(%arg1, %arg11, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
          %31 = polygeist.submap(%30, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%29, %31, %25 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%26, %27 : memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64):
            %32 = arith.mulf %in_16, %in : f64
            %33 = arith.addf %out, %32 : f64
            %34 = arith.mulf %in_16, %in_15 : f64
            %35 = arith.addf %out_17, %34 : f64
            linalg.yield %33, %35 : f64, f64
          }
        }
        %12 = polygeist.submap(%alloca_13, %c5, %c5, %c5) {map = #map10} : (memref<5x5x2xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map11} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%arg4, %arg10, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
        %15 = polygeist.submap(%14, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %12 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%13 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %out: f64):
          %20 = arith.mulf %in_15, %in : f64
          %21 = arith.addf %out, %20 : f64
          linalg.yield %21 : f64
        }
        %16 = polygeist.submap(%alloca_13, %c5, %c5, %c5) {map = #map14} : (memref<5x5x2xf64>, index, index, index) -> memref<?x?x?xf64>
        %17 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map15} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %18 = polygeist.submap(%arg1, %arg10, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
        %19 = polygeist.submap(%18, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %16 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%17 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %out: f64):
          %20 = arith.mulf %in_15, %in : f64
          %21 = arith.subf %out, %20 : f64
          linalg.yield %21 : f64
        }
      }
      affine.for %arg10 = 0 to 4 {
        %10 = polygeist.submap(%alloca_11, %c5, %c5) {map = #map1} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%10 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %11 = polygeist.submap(%alloca_11, %c5, %c5) {map = #map3} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%11 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 4 {
          %20 = polygeist.submap(%alloca_10, %c5) {map = #map4} : (memref<5xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%20 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %21 = polygeist.submap(%arg0, %c3, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %22 = polygeist.submap(%alloca_10, %c3, %c5) {map = #map6} : (memref<5xf64>, index, index) -> memref<?x?xf64>
          %23 = polygeist.submap(%arg7, %arg10, %arg11, %arg9, %c3) {map = #map16} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          %24 = polygeist.submap(%23, %c3, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction", "parallel"]} ins(%24, %21 : memref<?x?xf64>, memref<?x?xf64>) outs(%22 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %32 = arith.mulf %in, %in_15 : f64
            %33 = arith.addf %out, %32 : f64
            linalg.yield %33 : f64
          }
          %25 = polygeist.submap(%alloca_10, %c5, %c5) {map = #map6} : (memref<5xf64>, index, index) -> memref<?x?xf64>
          %26 = polygeist.submap(%alloca_11, %c5, %c5) {map = #map17} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
          %27 = polygeist.submap(%alloca_11, %c5, %c5) {map = #map18} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
          %28 = polygeist.submap(%arg4, %arg11, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
          %29 = polygeist.submap(%28, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %30 = polygeist.submap(%arg1, %arg11, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
          %31 = polygeist.submap(%30, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%29, %31, %25 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%26, %27 : memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64):
            %32 = arith.mulf %in, %in_16 : f64
            %33 = arith.addf %out, %32 : f64
            %34 = arith.mulf %in_15, %in_16 : f64
            %35 = arith.addf %out_17, %34 : f64
            linalg.yield %33, %35 : f64, f64
          }
        }
        %12 = polygeist.submap(%alloca_11, %c5, %c5, %c5) {map = #map10} : (memref<5x5x2xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map19} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%arg4, %arg10, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
        %15 = polygeist.submap(%14, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %12 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%13 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %out: f64):
          %20 = arith.mulf %in_15, %in : f64
          %21 = arith.subf %out, %20 : f64
          linalg.yield %21 : f64
        }
        %16 = polygeist.submap(%alloca_11, %c5, %c5, %c5) {map = #map14} : (memref<5x5x2xf64>, index, index, index) -> memref<?x?x?xf64>
        %17 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map15} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %18 = polygeist.submap(%arg1, %arg10, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
        %19 = polygeist.submap(%18, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %16 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%17 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %out: f64):
          %20 = arith.mulf %in_15, %in : f64
          %21 = arith.addf %out, %20 : f64
          linalg.yield %21 : f64
        }
      }
      affine.for %arg10 = 0 to 4 {
        %10 = polygeist.submap(%alloca_9, %c5, %c5) {map = #map1} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%10 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %11 = polygeist.submap(%alloca_9, %c5, %c5) {map = #map3} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%11 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 4 {
          %20 = polygeist.submap(%alloca_8, %c5) {map = #map4} : (memref<5xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%20 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %21 = polygeist.submap(%arg0, %c3, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %22 = polygeist.submap(%alloca_8, %c3, %c5) {map = #map6} : (memref<5xf64>, index, index) -> memref<?x?xf64>
          %23 = polygeist.submap(%arg7, %arg10, %arg11, %arg9, %c3) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          %24 = polygeist.submap(%23, %c3, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction", "parallel"]} ins(%24, %21 : memref<?x?xf64>, memref<?x?xf64>) outs(%22 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %32 = arith.mulf %in, %in_15 : f64
            %33 = arith.addf %out, %32 : f64
            linalg.yield %33 : f64
          }
          %25 = polygeist.submap(%alloca_8, %c5, %c5) {map = #map6} : (memref<5xf64>, index, index) -> memref<?x?xf64>
          %26 = polygeist.submap(%alloca_9, %c5, %c5) {map = #map17} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
          %27 = polygeist.submap(%alloca_9, %c5, %c5) {map = #map18} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
          %28 = polygeist.submap(%arg1, %arg11, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
          %29 = polygeist.submap(%28, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %30 = polygeist.submap(%arg4, %arg11, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
          %31 = polygeist.submap(%30, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%29, %31, %25 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%26, %27 : memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64):
            %32 = arith.mulf %in_16, %in : f64
            %33 = arith.addf %out, %32 : f64
            %34 = arith.mulf %in_16, %in_15 : f64
            %35 = arith.addf %out_17, %34 : f64
            linalg.yield %33, %35 : f64, f64
          }
        }
        %12 = polygeist.submap(%alloca_9, %c5, %c5, %c5) {map = #map21} : (memref<5x5x2xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map22} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%arg1, %arg10, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
        %15 = polygeist.submap(%14, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %12 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%13 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %out: f64):
          %20 = arith.mulf %in_15, %in : f64
          %21 = arith.addf %out, %20 : f64
          linalg.yield %21 : f64
        }
        %16 = polygeist.submap(%alloca_9, %c5, %c5, %c5) {map = #map23} : (memref<5x5x2xf64>, index, index, index) -> memref<?x?x?xf64>
        %17 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map24} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %18 = polygeist.submap(%arg4, %arg10, %c5) {map = #map9} : (memref<?xf64>, index, index) -> memref<?xf64>
        %19 = polygeist.submap(%18, %c5, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %16 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%17 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %out: f64):
          %20 = arith.mulf %in_15, %in : f64
          %21 = arith.subf %out, %20 : f64
          linalg.yield %21 : f64
        }
      }
      %1 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map25} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %2 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map26} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %3 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map27} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %4 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map28} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %5 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map29} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %6 = polygeist.submap(%arg6, %arg9, %c5, %c5, %c5) {map = #map30} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %7 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map19} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
      %8 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map11} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
      %9 = polygeist.submap(%alloca_14, %c5, %c5, %c5) {map = #map15} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map13, #map13, #map13, #map13, #map13, #map13, #map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %2, %3, %4, %5, %6 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%7, %8, %9 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) {
      ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %in_19: f64, %out: f64, %out_20: f64, %out_21: f64):
        %10 = arith.mulf %in, %out : f64
        %11 = arith.mulf %in_15, %out_20 : f64
        %12 = arith.addf %10, %11 : f64
        %13 = arith.mulf %in_16, %out_21 : f64
        %14 = arith.addf %12, %13 : f64
        %15 = arith.mulf %in_15, %out : f64
        %16 = arith.mulf %in_17, %out_20 : f64
        %17 = arith.addf %15, %16 : f64
        %18 = arith.mulf %in_18, %out_21 : f64
        %19 = arith.addf %17, %18 : f64
        %20 = arith.mulf %in_16, %out : f64
        %21 = arith.mulf %in_18, %out_20 : f64
        %22 = arith.addf %20, %21 : f64
        %23 = arith.mulf %in_19, %out_21 : f64
        %24 = arith.addf %22, %23 : f64
        linalg.yield %14, %19, %24 : f64, f64, f64
      }
      affine.for %arg10 = 0 to 5 {
        %10 = polygeist.submap(%alloca_6, %c4, %c3) {map = #map2} : (memref<4x3xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%10 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %11 = polygeist.submap(%alloca_7, %c4, %c3) {map = #map2} : (memref<4x3xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%11 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 5 {
          %17 = polygeist.submap(%alloca_5, %c3) {map = #map31} : (memref<3x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%17 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %18 = polygeist.submap(%alloca_5, %c3) {map = #map32} : (memref<3x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%18 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg12 = 0 to 5 {
            %27 = affine.load %alloca_14[%arg10, %arg11, %arg12, 1] : memref<5x5x5x3xf64>
            %28 = affine.load %alloca_14[%arg10, %arg11, %arg12, 2] : memref<5x5x5x3xf64>
            %29 = polygeist.submap(%arg2, %arg12, %c3) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
            %30 = polygeist.submap(%alloca_5, %c3) {map = #map32} : (memref<3x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%29 : memref<?xf64>) outs(%30 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %33 = arith.mulf %in, %27 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
            %31 = polygeist.submap(%arg2, %arg12, %c3) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
            %32 = polygeist.submap(%alloca_5, %c3) {map = #map31} : (memref<3x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%31 : memref<?xf64>) outs(%32 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %33 = arith.mulf %in, %28 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
          }
          %19 = polygeist.submap(%alloca_5, %c4, %c3) {map = #map34} : (memref<3x2xf64>, index, index) -> memref<?x?xf64>
          %20 = polygeist.submap(%alloca_6, %c4, %c3) {map = #map2} : (memref<4x3xf64>, index, index) -> memref<?x?xf64>
          %21 = polygeist.submap(%arg3, %arg11, %c4) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
          %22 = polygeist.submap(%21, %c4, %c3) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%22, %19 : memref<?x?xf64>, memref<?x?xf64>) outs(%20 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %27 = arith.mulf %in_15, %in : f64
            %28 = arith.addf %out, %27 : f64
            linalg.yield %28 : f64
          }
          %23 = polygeist.submap(%alloca_5, %c4, %c3) {map = #map35} : (memref<3x2xf64>, index, index) -> memref<?x?xf64>
          %24 = polygeist.submap(%alloca_7, %c4, %c3) {map = #map2} : (memref<4x3xf64>, index, index) -> memref<?x?xf64>
          %25 = polygeist.submap(%arg5, %arg11, %c4) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
          %26 = polygeist.submap(%25, %c4, %c3) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26, %23 : memref<?x?xf64>, memref<?x?xf64>) outs(%24 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %27 = arith.mulf %in_15, %in : f64
            %28 = arith.addf %out, %27 : f64
            linalg.yield %28 : f64
          }
        }
        %12 = polygeist.submap(%alloca_6, %c4, %c4, %c3) {map = #map36} : (memref<4x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%arg5, %arg10, %c4, %c4, %c3) {map = #map37} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%alloca_7, %c4, %c4, %c3) {map = #map36} : (memref<4x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %15 = polygeist.submap(%arg3, %arg10, %c4, %c4, %c3) {map = #map37} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %16 = polygeist.submap(%arg8, %arg9, %c4, %c4, %c3) {map = #map38} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %13, %14, %15 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%16 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
          %17 = arith.mulf %in, %in_15 : f64
          %18 = arith.mulf %in_16, %in_17 : f64
          %19 = arith.subf %17, %18 : f64
          %20 = arith.addf %out, %19 : f64
          linalg.yield %20 : f64
        }
      }
      affine.for %arg10 = 0 to 5 {
        %10 = polygeist.submap(%alloca_3, %c3, %c4) {map = #map2} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%10 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %11 = polygeist.submap(%alloca_4, %c3, %c4) {map = #map2} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%11 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 5 {
          %17 = polygeist.submap(%alloca_2, %c3) {map = #map31} : (memref<3x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%17 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %18 = polygeist.submap(%alloca_2, %c3) {map = #map32} : (memref<3x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%18 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg12 = 0 to 5 {
            %27 = affine.load %alloca_14[%arg10, %arg12, %arg11, 2] : memref<5x5x5x3xf64>
            %28 = affine.load %alloca_14[%arg10, %arg12, %arg11, 0] : memref<5x5x5x3xf64>
            %29 = polygeist.submap(%arg2, %arg12, %c3) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
            %30 = polygeist.submap(%alloca_2, %c3) {map = #map32} : (memref<3x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%29 : memref<?xf64>) outs(%30 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %33 = arith.mulf %in, %27 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
            %31 = polygeist.submap(%arg2, %arg12, %c3) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
            %32 = polygeist.submap(%alloca_2, %c3) {map = #map31} : (memref<3x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%31 : memref<?xf64>) outs(%32 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %33 = arith.mulf %in, %28 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
          }
          %19 = polygeist.submap(%alloca_2, %c4, %c3) {map = #map34} : (memref<3x2xf64>, index, index) -> memref<?x?xf64>
          %20 = polygeist.submap(%alloca_4, %c4, %c3) {map = #map39} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
          %21 = polygeist.submap(%arg5, %arg11, %c4) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
          %22 = polygeist.submap(%21, %c4, %c3) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%22, %19 : memref<?x?xf64>, memref<?x?xf64>) outs(%20 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %27 = arith.mulf %in_15, %in : f64
            %28 = arith.addf %out, %27 : f64
            linalg.yield %28 : f64
          }
          %23 = polygeist.submap(%alloca_2, %c4, %c3) {map = #map35} : (memref<3x2xf64>, index, index) -> memref<?x?xf64>
          %24 = polygeist.submap(%alloca_3, %c4, %c3) {map = #map39} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
          %25 = polygeist.submap(%arg3, %arg11, %c4) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
          %26 = polygeist.submap(%25, %c4, %c3) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26, %23 : memref<?x?xf64>, memref<?x?xf64>) outs(%24 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %27 = arith.mulf %in_15, %in : f64
            %28 = arith.addf %out, %27 : f64
            linalg.yield %28 : f64
          }
        }
        %12 = polygeist.submap(%alloca_3, %c4, %c3, %c4) {map = #map36} : (memref<3x4xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%arg5, %arg10, %c4, %c3, %c4) {map = #map37} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%alloca_4, %c4, %c3, %c4) {map = #map36} : (memref<3x4xf64>, index, index, index) -> memref<?x?x?xf64>
        %15 = polygeist.submap(%arg3, %arg10, %c4, %c3, %c4) {map = #map37} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %16 = polygeist.submap(%arg8, %arg9, %c4, %c3, %c4) {map = #map40} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %13, %14, %15 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%16 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
          %17 = arith.negf %in : f64
          %18 = arith.mulf %17, %in_15 : f64
          %19 = arith.mulf %in_16, %in_17 : f64
          %20 = arith.addf %18, %19 : f64
          %21 = arith.addf %out, %20 : f64
          linalg.yield %21 : f64
        }
      }
      affine.for %arg10 = 0 to 5 {
        %10 = polygeist.submap(%alloca_0, %c3, %c4) {map = #map2} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%10 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %11 = polygeist.submap(%alloca_1, %c3, %c4) {map = #map2} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%11 : memref<?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg11 = 0 to 5 {
          %17 = polygeist.submap(%alloca, %c3) {map = #map31} : (memref<3x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%17 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %18 = polygeist.submap(%alloca, %c3) {map = #map32} : (memref<3x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%18 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg12 = 0 to 5 {
            %27 = affine.load %alloca_14[%arg12, %arg11, %arg10, 0] : memref<5x5x5x3xf64>
            %28 = affine.load %alloca_14[%arg12, %arg11, %arg10, 1] : memref<5x5x5x3xf64>
            %29 = polygeist.submap(%arg2, %arg12, %c3) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
            %30 = polygeist.submap(%alloca, %c3) {map = #map32} : (memref<3x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%29 : memref<?xf64>) outs(%30 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %33 = arith.mulf %in, %27 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
            %31 = polygeist.submap(%arg2, %arg12, %c3) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
            %32 = polygeist.submap(%alloca, %c3) {map = #map31} : (memref<3x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%31 : memref<?xf64>) outs(%32 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %33 = arith.mulf %in, %28 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
          }
          %19 = polygeist.submap(%alloca, %c4, %c3) {map = #map35} : (memref<3x2xf64>, index, index) -> memref<?x?xf64>
          %20 = polygeist.submap(%alloca_1, %c4, %c3) {map = #map39} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
          %21 = polygeist.submap(%arg3, %arg11, %c4) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
          %22 = polygeist.submap(%21, %c4, %c3) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%22, %19 : memref<?x?xf64>, memref<?x?xf64>) outs(%20 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %27 = arith.mulf %in, %in_15 : f64
            %28 = arith.addf %out, %27 : f64
            linalg.yield %28 : f64
          }
          %23 = polygeist.submap(%alloca, %c4, %c3) {map = #map34} : (memref<3x2xf64>, index, index) -> memref<?x?xf64>
          %24 = polygeist.submap(%alloca_0, %c4, %c3) {map = #map39} : (memref<3x4xf64>, index, index) -> memref<?x?xf64>
          %25 = polygeist.submap(%arg5, %arg11, %c4) {map = #map33} : (memref<?xf64>, index, index) -> memref<?xf64>
          %26 = polygeist.submap(%25, %c4, %c3) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26, %23 : memref<?x?xf64>, memref<?x?xf64>) outs(%24 : memref<?x?xf64>) {
          ^bb0(%in: f64, %in_15: f64, %out: f64):
            %27 = arith.mulf %in, %in_15 : f64
            %28 = arith.addf %out, %27 : f64
            linalg.yield %28 : f64
          }
        }
        %12 = polygeist.submap(%alloca_0, %c4, %c4, %c3) {map = #map41} : (memref<3x4xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%arg3, %arg10, %c4, %c4, %c3) {map = #map37} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%alloca_1, %c4, %c4, %c3) {map = #map41} : (memref<3x4xf64>, index, index, index) -> memref<?x?x?xf64>
        %15 = polygeist.submap(%arg5, %arg10, %c4, %c4, %c3) {map = #map37} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %16 = polygeist.submap(%arg8, %arg9, %c4, %c4, %c3) {map = #map42} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map13, #map13, #map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %13, %14, %15 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%16 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
          %17 = arith.mulf %in, %in_15 : f64
          %18 = arith.mulf %in_16, %in_17 : f64
          %19 = arith.subf %17, %18 : f64
          %20 = arith.addf %out, %19 : f64
          linalg.yield %20 : f64
        }
      }
    }
    return
  }
}
