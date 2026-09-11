#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (d0, 1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> (d0, 0)>
#map5 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map6 = affine_map<(d0, d1) -> (d1, 1)>
#map7 = affine_map<(d0, d1) -> (d1, 0)>
#map8 = affine_map<(d0, d1) -> (d0, d1, 0)>
#map9 = affine_map<(d0, d1) -> (d0, d1, 1)>
#map10 = affine_map<(d0, d1) -> (d0, d1, 2)>
#map11 = affine_map<(d0, d1) -> (d0)>
#map12 = affine_map<(d0, d1) -> (d0, d1)>
#map13 = affine_map<(d0, d1)[s0] -> (s0, d0, d1, 0)>
#map14 = affine_map<(d0, d1)[s0] -> (s0, d0, d1, 1)>
#map15 = affine_map<(d0, d1)[s0] -> (s0, d0, d1, 2)>
#map16 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + d1 * 5 + s0 * 750)>
#map17 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 125)>
#map18 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 250)>
#map19 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 375)>
#map20 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 500)>
#map21 = affine_map<(d0, d1, d2)[s0] -> (d2 + d0 * 25 + s0 * 750 + d1 * 5 + 625)>
#map22 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 0)>
#map23 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 1)>
#map24 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 2)>
#map25 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map26 = affine_map<(d0) -> (d0, 2)>
#map27 = affine_map<(d0)[s0] -> (s0, d0, 0)>
#map28 = affine_map<(d0)[s0] -> (s0, d0, 1)>
#map29 = affine_map<(d0)[s0] -> (s0, d0, 2)>
#map30 = affine_map<(d0, d1, d2) -> (d1, d2, 0)>
#map31 = affine_map<(d0, d1, d2) -> (d1, d2, 1)>
#map32 = affine_map<(d0, d1, d2)[s0] -> (d0 * 5 + s0)>
#map33 = affine_map<(d0, d1, d2) -> (d1, d2, 2)>
#map34 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 64 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x3xf64>
    %alloca_0 = memref.alloca() : memref<4x4x3xf64>
    %alloca_1 = memref.alloca() : memref<5x2xf64>
    %alloca_2 = memref.alloca() : memref<5x5x3xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5x3xf64>
    affine.for %arg7 = 0 to 2 {
      %0 = polygeist.submap(%alloca_3, %c5, %c5, %c5, %c3) {map = #map} : (memref<5x5x5x3xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg8 = 0 to 4 {
        %10 = polygeist.submap(%alloca_2, %c5, %c5, %c3) {map = #map1} : (memref<5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%10 : memref<?x?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 4 {
          %11 = polygeist.submap(%alloca_1, %c5) {map = #map2} : (memref<5x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%11 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %12 = polygeist.submap(%alloca_1, %c5) {map = #map4} : (memref<5x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%12 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg10 = 0 to 4 {
            %22 = affine.load %arg5[%arg7 * 64 + %arg10 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
            %23 = polygeist.submap(%arg0, %arg10, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?xf64>
            %24 = polygeist.submap(%alloca_1, %c5) {map = #map4} : (memref<5x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%23 : memref<?xf64>) outs(%24 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %27 = arith.mulf %22, %in : f64
              %28 = arith.addf %out, %27 : f64
              linalg.yield %28 : f64
            }
            %25 = polygeist.submap(%arg1, %arg10, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?xf64>
            %26 = polygeist.submap(%alloca_1, %c5) {map = #map2} : (memref<5x2xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%25 : memref<?xf64>) outs(%26 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %27 = arith.mulf %22, %in : f64
              %28 = arith.addf %out, %27 : f64
              linalg.yield %28 : f64
            }
          }
          %13 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map6} : (memref<5x2xf64>, index, index) -> memref<?x?xf64>
          %14 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map7} : (memref<5x2xf64>, index, index) -> memref<?x?xf64>
          %15 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map8} : (memref<5x5x3xf64>, index, index) -> memref<?x?xf64>
          %16 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map9} : (memref<5x5x3xf64>, index, index) -> memref<?x?xf64>
          %17 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map10} : (memref<5x5x3xf64>, index, index) -> memref<?x?xf64>
          %18 = polygeist.submap(%arg0, %arg9, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?xf64>
          %19 = polygeist.submap(%18, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %20 = polygeist.submap(%arg1, %arg9, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?xf64>
          %21 = polygeist.submap(%20, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map12, #map12, #map12, #map12, #map12, #map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%19, %21, %13, %14 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%15, %16, %17 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64, %out_7: f64, %out_8: f64):
            %22 = arith.mulf %in_5, %in : f64
            %23 = arith.addf %out, %22 : f64
            %24 = arith.mulf %in_6, %in_4 : f64
            %25 = arith.addf %out_7, %24 : f64
            %26 = arith.mulf %in_6, %in : f64
            %27 = arith.addf %out_8, %26 : f64
            linalg.yield %23, %25, %27 : f64, f64, f64
          }
        }
        affine.for %arg9 = 0 to 5 {
          %11 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
          %12 = affine.load %arg1[%arg8 + %arg9 * 4] : memref<?xf64>
          %13 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map8} : (memref<5x5x3xf64>, index, index) -> memref<?x?xf64>
          %14 = polygeist.submap(%alloca_3, %arg9, %c5, %c5) {map = #map13} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%13 : memref<?x?xf64>) outs(%14 : memref<?x?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %19 = arith.mulf %in, %11 : f64
            %20 = arith.addf %out, %19 : f64
            linalg.yield %20 : f64
          }
          %15 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map9} : (memref<5x5x3xf64>, index, index) -> memref<?x?xf64>
          %16 = polygeist.submap(%alloca_3, %arg9, %c5, %c5) {map = #map14} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%15 : memref<?x?xf64>) outs(%16 : memref<?x?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %19 = arith.mulf %in, %11 : f64
            %20 = arith.addf %out, %19 : f64
            linalg.yield %20 : f64
          }
          %17 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map10} : (memref<5x5x3xf64>, index, index) -> memref<?x?xf64>
          %18 = polygeist.submap(%alloca_3, %arg9, %c5, %c5) {map = #map15} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%17 : memref<?x?xf64>) outs(%18 : memref<?x?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %19 = arith.mulf %in, %12 : f64
            %20 = arith.addf %out, %19 : f64
            linalg.yield %20 : f64
          }
        } {polygeist.was_parallel}
      }
      %1 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map16} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %2 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %3 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map18} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %4 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %5 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %6 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map21} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %7 = polygeist.submap(%alloca_3, %c5, %c5, %c5) {map = #map22} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
      %8 = polygeist.submap(%alloca_3, %c5, %c5, %c5) {map = #map23} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
      %9 = polygeist.submap(%alloca_3, %c5, %c5, %c5) {map = #map24} : (memref<5x5x5x3xf64>, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %2, %3, %4, %5, %6 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%7, %8, %9 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) {
      ^bb0(%in: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64, %out_9: f64, %out_10: f64):
        %10 = arith.mulf %in, %out : f64
        %11 = arith.mulf %in_4, %out_9 : f64
        %12 = arith.addf %10, %11 : f64
        %13 = arith.mulf %in_5, %out_10 : f64
        %14 = arith.addf %12, %13 : f64
        %15 = arith.mulf %in_4, %out : f64
        %16 = arith.mulf %in_6, %out_9 : f64
        %17 = arith.addf %15, %16 : f64
        %18 = arith.mulf %in_7, %out_10 : f64
        %19 = arith.addf %17, %18 : f64
        %20 = arith.mulf %in_5, %out : f64
        %21 = arith.mulf %in_7, %out_9 : f64
        %22 = arith.addf %20, %21 : f64
        %23 = arith.mulf %in_8, %out_10 : f64
        %24 = arith.addf %22, %23 : f64
        linalg.yield %14, %19, %24 : f64, f64, f64
      }
      affine.for %arg8 = 0 to 5 {
        %10 = polygeist.submap(%alloca_0, %c4, %c4, %c3) {map = #map1} : (memref<4x4x3xf64>, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%10 : memref<?x?x?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 5 {
          %17 = polygeist.submap(%alloca, %c4, %c3) {map = #map12} : (memref<4x3xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map12], iterator_types = ["parallel", "parallel"]} outs(%17 : memref<?x?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          affine.for %arg10 = 0 to 5 {
            %18 = affine.load %alloca_3[%arg8, %arg9, %arg10, 0] : memref<5x5x5x3xf64>
            %19 = affine.load %alloca_3[%arg8, %arg9, %arg10, 1] : memref<5x5x5x3xf64>
            %20 = affine.load %alloca_3[%arg8, %arg9, %arg10, 2] : memref<5x5x5x3xf64>
            %21 = polygeist.submap(%arg3, %arg10, %c4) {map = #map25} : (memref<?xf64>, index, index) -> memref<?xf64>
            %22 = polygeist.submap(%alloca, %c4) {map = #map4} : (memref<4x3xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%21 : memref<?xf64>) outs(%22 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %27 = arith.mulf %18, %in : f64
              %28 = arith.addf %out, %27 : f64
              linalg.yield %28 : f64
            }
            %23 = polygeist.submap(%arg2, %arg10, %c4) {map = #map25} : (memref<?xf64>, index, index) -> memref<?xf64>
            %24 = polygeist.submap(%alloca, %c4) {map = #map2} : (memref<4x3xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%23 : memref<?xf64>) outs(%24 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %27 = arith.mulf %19, %in : f64
              %28 = arith.addf %out, %27 : f64
              linalg.yield %28 : f64
            }
            %25 = polygeist.submap(%arg2, %arg10, %c4) {map = #map25} : (memref<?xf64>, index, index) -> memref<?xf64>
            %26 = polygeist.submap(%alloca, %c4) {map = #map26} : (memref<4x3xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%25 : memref<?xf64>) outs(%26 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %27 = arith.mulf %20, %in : f64
              %28 = arith.addf %out, %27 : f64
              linalg.yield %28 : f64
            }
          }
          affine.for %arg10 = 0 to 4 {
            %18 = affine.load %arg2[%arg9 + %arg10 * 5] : memref<?xf64>
            %19 = affine.load %arg3[%arg9 + %arg10 * 5] : memref<?xf64>
            %20 = polygeist.submap(%alloca, %c4) {map = #map4} : (memref<4x3xf64>, index) -> memref<?xf64>
            %21 = polygeist.submap(%alloca_0, %arg10, %c4) {map = #map27} : (memref<4x4x3xf64>, index, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%20 : memref<?xf64>) outs(%21 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %26 = arith.mulf %in, %18 : f64
              %27 = arith.addf %out, %26 : f64
              linalg.yield %27 : f64
            }
            %22 = polygeist.submap(%alloca, %c4) {map = #map2} : (memref<4x3xf64>, index) -> memref<?xf64>
            %23 = polygeist.submap(%alloca_0, %arg10, %c4) {map = #map28} : (memref<4x4x3xf64>, index, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%22 : memref<?xf64>) outs(%23 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %26 = arith.mulf %in, %19 : f64
              %27 = arith.addf %out, %26 : f64
              linalg.yield %27 : f64
            }
            %24 = polygeist.submap(%alloca, %c4) {map = #map26} : (memref<4x3xf64>, index) -> memref<?xf64>
            %25 = polygeist.submap(%alloca_0, %arg10, %c4) {map = #map29} : (memref<4x4x3xf64>, index, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%24 : memref<?xf64>) outs(%25 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %26 = arith.mulf %in, %18 : f64
              %27 = arith.addf %out, %26 : f64
              linalg.yield %27 : f64
            }
          } {polygeist.was_parallel}
        }
        %11 = polygeist.submap(%alloca_0, %c4, %c4, %c4) {map = #map30} : (memref<4x4x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %12 = polygeist.submap(%alloca_0, %c4, %c4, %c4) {map = #map31} : (memref<4x4x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %13 = polygeist.submap(%arg2, %arg8, %c4, %c4, %c4) {map = #map32} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %14 = polygeist.submap(%alloca_0, %c4, %c4, %c4) {map = #map33} : (memref<4x4x3xf64>, index, index, index) -> memref<?x?x?xf64>
        %15 = polygeist.submap(%arg3, %arg8, %c4, %c4, %c4) {map = #map32} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        %16 = polygeist.submap(%arg6, %arg7, %c4, %c4, %c4) {map = #map34} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %12, %13, %14, %15 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%16 : memref<?x?x?xf64>) {
        ^bb0(%in: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %out: f64):
          %17 = arith.addf %in, %in_4 : f64
          %18 = arith.mulf %17, %in_5 : f64
          %19 = arith.mulf %in_6, %in_7 : f64
          %20 = arith.addf %18, %19 : f64
          %21 = arith.addf %out, %20 : f64
          linalg.yield %21 : f64
        }
      }
    }
    return
  }
}
