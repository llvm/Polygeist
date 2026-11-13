#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d1 * 25 + s0 * 375 + s1 + s2 * 5)>
#map2 = affine_map<(d0, d1) -> (d1 * 4 + d0)>
#map3 = affine_map<(d0, d1)[s0, s1, s2] -> (d1 * 25 + s0 * 375 + s1 + s2 * 5 + 125)>
#map4 = affine_map<(d0, d1)[s0, s1, s2] -> (d1 * 25 + s0 * 375 + s1 + s2 * 5 + 250)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0)[s0, s1] -> (s0, s1, d0)>
#map8 = affine_map<(d0, d1)[s0] -> (s0, d1, d0)>
#map9 = affine_map<(d0, d1)[s0] -> (d1 * 4 + s0)>
#map10 = affine_map<(d0, d1)[s0] -> (d1, s0, d0)>
#map11 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 64 + s1 * 16 + s2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_grad_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<5x4x4xf64>
    %alloca_1 = memref.alloca() : memref<5x4x4xf64>
    %alloca_2 = memref.alloca() : memref<5x5x4xf64>
    %alloca_3 = memref.alloca() : memref<5x5x4xf64>
    %alloca_4 = memref.alloca() : memref<5x5x4xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          %alloca_5 = memref.alloca(%c4) : memref<?xf64>
          %alloca_6 = memref.alloca(%c4) : memref<?xf64>
          %alloca_7 = memref.alloca(%c4) : memref<?xf64>
          %0 = polygeist.submap(%alloca_5, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %1 = polygeist.submap(%alloca_6, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%alloca_7, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %3 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c4, %c5) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?xf64>
          %4 = polygeist.submap(%arg2, %c4, %c5) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %5 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c4, %c5) {map = #map3} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%arg1, %c4, %c5) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c4, %c5) {map = #map4} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_5, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %9 = polygeist.submap(%alloca_6, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_7, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%3, %4, %5, %6, %7 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%8, %9, %10 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %out: f64, %out_12: f64, %out_13: f64):
            %17 = arith.mulf %in, %in_8 : f64
            %18 = arith.addf %out_13, %17 : f64
            %19 = arith.mulf %in_9, %in_10 : f64
            %20 = arith.addf %out_12, %19 : f64
            %21 = arith.mulf %in_11, %in_10 : f64
            %22 = arith.addf %out, %21 : f64
            linalg.yield %22, %20, %18 : f64, f64, f64
          }
          %11 = polygeist.submap(%alloca_7, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %12 = polygeist.submap(%alloca_4, %arg5, %arg6, %c4) {map = #map7} : (memref<5x5x4xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%11 : memref<?xf64>) outs(%12 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %13 = polygeist.submap(%alloca_6, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %14 = polygeist.submap(%alloca_3, %arg5, %arg6, %c4) {map = #map7} : (memref<5x5x4xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : memref<?xf64>) outs(%14 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %15 = polygeist.submap(%alloca_5, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %16 = polygeist.submap(%alloca_2, %arg5, %arg6, %c4) {map = #map7} : (memref<5x5x4xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%15 : memref<?xf64>) outs(%16 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          %alloca_5 = memref.alloca(%c4) : memref<?xf64>
          %alloca_6 = memref.alloca(%c4) : memref<?xf64>
          %alloca_7 = memref.alloca(%c4) : memref<?xf64>
          %0 = polygeist.submap(%alloca_5, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %1 = polygeist.submap(%alloca_6, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%alloca_7, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %3 = polygeist.submap(%alloca_4, %arg5, %c4, %c5) {map = #map8} : (memref<5x5x4xf64>, index, index, index) -> memref<?x?xf64>
          %4 = polygeist.submap(%arg1, %arg6, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %5 = polygeist.submap(%alloca_3, %arg5, %c4, %c5) {map = #map8} : (memref<5x5x4xf64>, index, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%arg2, %arg6, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%alloca_2, %arg5, %c4, %c5) {map = #map8} : (memref<5x5x4xf64>, index, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_5, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %9 = polygeist.submap(%alloca_6, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_7, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%3, %4, %5, %6, %7 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%8, %9, %10 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %out: f64, %out_12: f64, %out_13: f64):
            %17 = arith.mulf %in, %in_8 : f64
            %18 = arith.addf %out_13, %17 : f64
            %19 = arith.mulf %in_9, %in_10 : f64
            %20 = arith.addf %out_12, %19 : f64
            %21 = arith.mulf %in_11, %in_8 : f64
            %22 = arith.addf %out, %21 : f64
            linalg.yield %22, %20, %18 : f64, f64, f64
          }
          %11 = polygeist.submap(%alloca_7, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %12 = polygeist.submap(%alloca_1, %arg5, %arg6, %c4) {map = #map7} : (memref<5x4x4xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%11 : memref<?xf64>) outs(%12 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %13 = polygeist.submap(%alloca_6, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %14 = polygeist.submap(%alloca_0, %arg5, %arg6, %c4) {map = #map7} : (memref<5x4x4xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : memref<?xf64>) outs(%14 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %15 = polygeist.submap(%alloca_5, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %16 = polygeist.submap(%alloca, %arg5, %arg6, %c4) {map = #map7} : (memref<5x4x4xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%15 : memref<?xf64>) outs(%16 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          %alloca_5 = memref.alloca(%c4) : memref<?xf64>
          %alloca_6 = memref.alloca(%c4) : memref<?xf64>
          %alloca_7 = memref.alloca(%c4) : memref<?xf64>
          %0 = polygeist.submap(%alloca_5, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %1 = polygeist.submap(%alloca_6, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%alloca_7, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %3 = polygeist.submap(%alloca_1, %arg6, %c4, %c5) {map = #map10} : (memref<5x4x4xf64>, index, index, index) -> memref<?x?xf64>
          %4 = polygeist.submap(%arg1, %arg5, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %5 = polygeist.submap(%alloca_0, %arg6, %c4, %c5) {map = #map10} : (memref<5x4x4xf64>, index, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%alloca, %arg6, %c4, %c5) {map = #map10} : (memref<5x4x4xf64>, index, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%arg2, %arg5, %c4, %c5) {map = #map9} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_5, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %9 = polygeist.submap(%alloca_6, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_7, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%3, %4, %5, %6, %7 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%8, %9, %10 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %out: f64, %out_12: f64, %out_13: f64):
            %15 = arith.mulf %in, %in_8 : f64
            %16 = arith.addf %out_13, %15 : f64
            %17 = arith.mulf %in_9, %in_8 : f64
            %18 = arith.addf %out_12, %17 : f64
            %19 = arith.mulf %in_10, %in_11 : f64
            %20 = arith.addf %out, %19 : f64
            linalg.yield %20, %18, %16 : f64, f64, f64
          }
          %11 = polygeist.submap(%alloca_5, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %12 = polygeist.submap(%alloca_6, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %13 = polygeist.submap(%alloca_7, %c4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %14 = polygeist.submap(%arg3, %arg4, %arg5, %arg6, %c4) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%11, %12, %13 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%14 : memref<?xf64>) {
          ^bb0(%in: f64, %in_8: f64, %in_9: f64, %out: f64):
            %15 = arith.addf %in_9, %in_8 : f64
            %16 = arith.addf %15, %in : f64
            %17 = arith.addf %out, %16 : f64
            linalg.yield %17 : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    }
    return
  }
}
