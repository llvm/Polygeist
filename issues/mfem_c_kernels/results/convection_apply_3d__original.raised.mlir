#map = affine_map<(d0)[s0, s1] -> (s0, s1, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1 + d0 * 4)>
#map3 = affine_map<(d0, d1)[s0, s1, s2] -> (d1 + s0 * 64 + s1 * 16 + s2 * 4)>
#map4 = affine_map<(d0, d1)[s0, s1] -> (s0, s1, d0)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0)[s0, s1] -> (s0, d0, s1)>
#map8 = affine_map<(d0, d1)[s0, s1] -> (s0, d1, s1)>
#map9 = affine_map<(d0, d1)[s0, s1] -> (s0, d0, s1)>
#map10 = affine_map<(d0)[s0, s1] -> (d0, s0, s1)>
#map11 = affine_map<(d0, d1)[s0, s1] -> (d1, s0, s1)>
#map12 = affine_map<(d0, d1)[s0, s1] -> (d0, s0, s1)>
#map13 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5)>
#map14 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5 + 125)>
#map15 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5 + 250)>
#map16 = affine_map<(d0, d1) -> (d1 + d0 * 5)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map19 = affine_map<(d0, d1, d2, d3)[s0] -> (d2 + s0 * 64 + d0 * 16 + d1 * 4)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x4x5xf64>
    %alloca_0 = memref.alloca() : memref<4x5x5xf64>
    %alloca_1 = memref.alloca() : memref<5x5x5xf64>
    %alloca_2 = memref.alloca() : memref<5x5x5xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5xf64>
    %alloca_4 = memref.alloca() : memref<5x5x5xf64>
    %alloca_5 = memref.alloca() : memref<4x5x5xf64>
    %alloca_6 = memref.alloca() : memref<4x5x5xf64>
    %alloca_7 = memref.alloca() : memref<4x5x5xf64>
    %alloca_8 = memref.alloca() : memref<4x4x5xf64>
    %alloca_9 = memref.alloca() : memref<4x4x5xf64>
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          %alloca_10 = memref.alloca(%c5) : memref<?xf64>
          %alloca_11 = memref.alloca(%c5) : memref<?xf64>
          %3 = polygeist.submap(%alloca_8, %arg7, %arg8, %c5) {map = #map} : (memref<4x4x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %4 = polygeist.submap(%alloca_9, %arg7, %arg8, %c5) {map = #map} : (memref<4x4x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %5 = polygeist.submap(%alloca_10, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%5 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %6 = polygeist.submap(%alloca_11, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%6 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %7 = polygeist.submap(%arg0, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%arg4, %arg6, %arg7, %arg8, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?xf64>
          %9 = polygeist.submap(%arg1, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_9, %arg7, %arg8, %c5, %c4) {map = #map4} : (memref<4x4x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %11 = polygeist.submap(%alloca_8, %arg7, %arg8, %c5, %c4) {map = #map4} : (memref<4x4x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %12 = polygeist.submap(%alloca_10, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %13 = polygeist.submap(%alloca_11, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%7, %8, %9 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%10, %11, %12, %13 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_12: f64, %in_13: f64, %out: f64, %out_14: f64, %out_15: f64, %out_16: f64):
            %14 = arith.mulf %in, %in_12 : f64
            %15 = arith.addf %out_16, %14 : f64
            %16 = arith.mulf %in_13, %in_12 : f64
            %17 = arith.addf %out_15, %16 : f64
            linalg.yield %15, %17, %17, %15 : f64, f64, f64, f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          %alloca_10 = memref.alloca(%c5) : memref<?xf64>
          %alloca_11 = memref.alloca(%c5) : memref<?xf64>
          %alloca_12 = memref.alloca(%c5) : memref<?xf64>
          %3 = polygeist.submap(%alloca_5, %arg7, %arg8, %c5) {map = #map7} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %4 = polygeist.submap(%alloca_6, %arg7, %arg8, %c5) {map = #map7} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %5 = polygeist.submap(%alloca_7, %arg7, %arg8, %c5) {map = #map7} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%5 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %6 = polygeist.submap(%alloca_10, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%6 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %7 = polygeist.submap(%alloca_11, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%7 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %8 = polygeist.submap(%alloca_12, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %9 = polygeist.submap(%arg0, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_9, %arg7, %arg8, %c5, %c4) {map = #map8} : (memref<4x4x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %11 = polygeist.submap(%arg1, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %12 = polygeist.submap(%arg0, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %13 = polygeist.submap(%alloca_8, %arg7, %arg8, %c5, %c4) {map = #map8} : (memref<4x4x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %14 = polygeist.submap(%alloca_7, %arg7, %arg8, %c5, %c4) {map = #map9} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %15 = polygeist.submap(%alloca_6, %arg7, %arg8, %c5, %c4) {map = #map9} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %16 = polygeist.submap(%alloca_5, %arg7, %arg8, %c5, %c4) {map = #map9} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %17 = polygeist.submap(%alloca_10, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %18 = polygeist.submap(%alloca_11, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %19 = polygeist.submap(%alloca_12, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%9, %10, %11, %12, %13 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%14, %15, %16, %17, %18, %19 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64, %out_18: f64, %out_19: f64, %out_20: f64, %out_21: f64):
            %20 = arith.mulf %in, %in_13 : f64
            %21 = arith.addf %out_21, %20 : f64
            %22 = arith.mulf %in_14, %in_13 : f64
            %23 = arith.addf %out_20, %22 : f64
            %24 = arith.mulf %in_15, %in_16 : f64
            %25 = arith.addf %out_19, %24 : f64
            linalg.yield %21, %23, %25, %25, %23, %21 : f64, f64, f64, f64, f64, f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          %alloca_10 = memref.alloca(%c5) : memref<?xf64>
          %alloca_11 = memref.alloca(%c5) : memref<?xf64>
          %alloca_12 = memref.alloca(%c5) : memref<?xf64>
          %3 = polygeist.submap(%alloca_2, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %4 = polygeist.submap(%alloca_3, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %5 = polygeist.submap(%alloca_4, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%5 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %6 = polygeist.submap(%alloca_10, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%6 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %7 = polygeist.submap(%alloca_11, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%7 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %8 = polygeist.submap(%alloca_12, %c5) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %9 = polygeist.submap(%arg1, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_7, %arg8, %arg7, %c5, %c4) {map = #map11} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %11 = polygeist.submap(%arg0, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %12 = polygeist.submap(%alloca_6, %arg8, %arg7, %c5, %c4) {map = #map11} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %13 = polygeist.submap(%arg0, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %14 = polygeist.submap(%alloca_5, %arg8, %arg7, %c5, %c4) {map = #map11} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %15 = polygeist.submap(%alloca_4, %arg8, %arg7, %c5, %c4) {map = #map12} : (memref<5x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %16 = polygeist.submap(%alloca_3, %arg8, %arg7, %c5, %c4) {map = #map12} : (memref<5x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %17 = polygeist.submap(%alloca_2, %arg8, %arg7, %c5, %c4) {map = #map12} : (memref<5x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %18 = polygeist.submap(%alloca_10, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %19 = polygeist.submap(%alloca_11, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %20 = polygeist.submap(%alloca_12, %c5, %c4) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%9, %10, %11, %12, %13, %14 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%15, %16, %17, %18, %19, %20 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64, %out_18: f64, %out_19: f64, %out_20: f64, %out_21: f64, %out_22: f64):
            %28 = arith.mulf %in, %in_13 : f64
            %29 = arith.addf %out_22, %28 : f64
            %30 = arith.mulf %in_14, %in_15 : f64
            %31 = arith.addf %out_21, %30 : f64
            %32 = arith.mulf %in_16, %in_17 : f64
            %33 = arith.addf %out_20, %32 : f64
            linalg.yield %29, %31, %33, %33, %31, %29 : f64, f64, f64, f64, f64, f64
          }
          %21 = polygeist.submap(%arg3, %arg6, %arg7, %arg8, %c5) {map = #map13} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          %22 = polygeist.submap(%alloca_2, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          %23 = polygeist.submap(%arg3, %arg6, %arg7, %arg8, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          %24 = polygeist.submap(%alloca_3, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          %25 = polygeist.submap(%arg3, %arg6, %arg7, %arg8, %c5) {map = #map15} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          %26 = polygeist.submap(%alloca_4, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          %27 = polygeist.submap(%alloca_1, %arg8, %arg7, %c5) {map = #map10} : (memref<5x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%21, %22, %23, %24, %25, %26 : memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%27 : memref<?xf64>) {
          ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
            %28 = arith.mulf %in, %in_13 : f64
            %29 = arith.mulf %in_14, %in_15 : f64
            %30 = arith.addf %28, %29 : f64
            %31 = arith.mulf %in_16, %in_17 : f64
            %32 = arith.addf %30, %31 : f64
            linalg.yield %32 : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          %alloca_10 = memref.alloca(%c4) : memref<?xf64>
          %3 = polygeist.submap(%alloca_0, %arg8, %arg7, %c4) {map = #map10} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %4 = polygeist.submap(%alloca_10, %c4) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %5 = polygeist.submap(%arg2, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%alloca_1, %arg8, %arg7, %c4, %c5) {map = #map11} : (memref<5x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%alloca_0, %arg8, %arg7, %c4, %c5) {map = #map12} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_10, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%5, %6 : memref<?x?xf64>, memref<?x?xf64>) outs(%7, %8 : memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_11: f64, %out: f64, %out_12: f64):
            %9 = arith.mulf %in, %in_11 : f64
            %10 = arith.addf %out_12, %9 : f64
            linalg.yield %10, %10 : f64, f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          %alloca_10 = memref.alloca(%c4) : memref<?xf64>
          %3 = polygeist.submap(%alloca, %arg7, %arg8, %c4) {map = #map7} : (memref<4x4x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %4 = polygeist.submap(%alloca_10, %c4) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %5 = polygeist.submap(%arg2, %c4, %c5) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%alloca_0, %arg7, %arg8, %c4, %c5) {map = #map8} : (memref<4x5x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%alloca, %arg7, %arg8, %c4, %c5) {map = #map9} : (memref<4x4x5xf64>, index, index, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_10, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%5, %6 : memref<?x?xf64>, memref<?x?xf64>) outs(%7, %8 : memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_11: f64, %out: f64, %out_12: f64):
            %9 = arith.mulf %in, %in_11 : f64
            %10 = arith.addf %out_12, %9 : f64
            linalg.yield %10, %10 : f64, f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      %0 = polygeist.submap(%arg2, %c4, %c4, %c4, %c5) {map = #map17} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %1 = polygeist.submap(%alloca, %c4, %c4, %c4, %c5) {map = #map18} : (memref<4x4x5xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %2 = polygeist.submap(%arg5, %arg6, %c4, %c4, %c4, %c5) {map = #map19} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map20, #map20, #map20], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%2 : memref<?x?x?x?xf64>) {
      ^bb0(%in: f64, %in_10: f64, %out: f64):
        %3 = arith.mulf %in, %in_10 : f64
        %4 = arith.addf %out, %3 : f64
        linalg.yield %4 : f64
      }
    }
    return
  }
}
