#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d1 + s0 * 64 + s1 * 16 + s2 * 4)>
#map2 = affine_map<(d0, d1) -> (d1 + d0 * 4)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0)[s0, s1] -> (s0, s1, d0)>
#map6 = affine_map<(d0, d1)[s0] -> (s0, d1, d0)>
#map7 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 4)>
#map8 = affine_map<(d0, d1)[s0] -> (d1, s0, d0)>
#map9 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5)>
#map10 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5 + 125)>
#map11 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5 + 250)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_interp_grad_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x5x5xf64>
    %alloca_0 = memref.alloca() : memref<4x5x5xf64>
    %alloca_1 = memref.alloca() : memref<4x5x5xf64>
    %alloca_2 = memref.alloca() : memref<4x4x5xf64>
    %alloca_3 = memref.alloca() : memref<4x4x5xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          %alloca_4 = memref.alloca(%c5) : memref<?xf64>
          %alloca_5 = memref.alloca(%c5) : memref<?xf64>
          %0 = polygeist.submap(%alloca_4, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %1 = polygeist.submap(%alloca_5, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?xf64>
          %3 = polygeist.submap(%arg1, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %4 = polygeist.submap(%arg2, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %5 = polygeist.submap(%alloca_4, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%alloca_5, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "reduction"]} ins(%2, %3, %4 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%5, %6 : memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_6: f64, %in_7: f64, %out: f64, %out_8: f64):
            %11 = arith.mulf %in, %in_6 : f64
            %12 = arith.addf %out_8, %11 : f64
            %13 = arith.mulf %in, %in_7 : f64
            %14 = arith.addf %out, %13 : f64
            linalg.yield %14, %12 : f64, f64
          }
          %7 = polygeist.submap(%alloca_5, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %8 = polygeist.submap(%alloca_3, %arg5, %arg6, %c5) {map = #map5} : (memref<4x4x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%7 : memref<?xf64>) outs(%8 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %9 = polygeist.submap(%alloca_4, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %10 = polygeist.submap(%alloca_2, %arg5, %arg6, %c5) {map = #map5} : (memref<4x4x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%9 : memref<?xf64>) outs(%10 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          %alloca_4 = memref.alloca(%c5) : memref<?xf64>
          %alloca_5 = memref.alloca(%c5) : memref<?xf64>
          %alloca_6 = memref.alloca(%c5) : memref<?xf64>
          %0 = polygeist.submap(%alloca_4, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %1 = polygeist.submap(%alloca_5, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%alloca_6, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %3 = polygeist.submap(%alloca_2, %arg5, %c5, %c4) {map = #map6} : (memref<4x4x5xf64>, index, index, index) -> memref<?x?xf64>
          %4 = polygeist.submap(%arg1, %arg6, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %5 = polygeist.submap(%alloca_3, %arg5, %c5, %c4) {map = #map6} : (memref<4x4x5xf64>, index, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%arg2, %arg6, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%alloca_4, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_5, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %9 = polygeist.submap(%alloca_6, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "reduction"]} ins(%3, %4, %5, %6 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%7, %8, %9 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64, %out_10: f64, %out_11: f64):
            %16 = arith.mulf %in, %in_7 : f64
            %17 = arith.addf %out_11, %16 : f64
            %18 = arith.mulf %in_8, %in_9 : f64
            %19 = arith.addf %out_10, %18 : f64
            %20 = arith.mulf %in_8, %in_7 : f64
            %21 = arith.addf %out, %20 : f64
            linalg.yield %21, %19, %17 : f64, f64, f64
          }
          %10 = polygeist.submap(%alloca_6, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %11 = polygeist.submap(%alloca_1, %arg5, %arg6, %c5) {map = #map5} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : memref<?xf64>) outs(%11 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %12 = polygeist.submap(%alloca_5, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %13 = polygeist.submap(%alloca_0, %arg5, %arg6, %c5) {map = #map5} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : memref<?xf64>) outs(%13 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %14 = polygeist.submap(%alloca_4, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %15 = polygeist.submap(%alloca, %arg5, %arg6, %c5) {map = #map5} : (memref<4x5x5xf64>, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : memref<?xf64>) outs(%15 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          %alloca_4 = memref.alloca(%c5) : memref<?xf64>
          %alloca_5 = memref.alloca(%c5) : memref<?xf64>
          %alloca_6 = memref.alloca(%c5) : memref<?xf64>
          %0 = polygeist.submap(%alloca_4, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %1 = polygeist.submap(%alloca_5, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %2 = polygeist.submap(%alloca_6, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %3 = polygeist.submap(%alloca_1, %arg6, %c5, %c4) {map = #map8} : (memref<4x5x5xf64>, index, index, index) -> memref<?x?xf64>
          %4 = polygeist.submap(%arg1, %arg5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %5 = polygeist.submap(%alloca_0, %arg6, %c5, %c4) {map = #map8} : (memref<4x5x5xf64>, index, index, index) -> memref<?x?xf64>
          %6 = polygeist.submap(%alloca, %arg6, %c5, %c4) {map = #map8} : (memref<4x5x5xf64>, index, index, index) -> memref<?x?xf64>
          %7 = polygeist.submap(%arg2, %arg5, %c5, %c4) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
          %8 = polygeist.submap(%alloca_4, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %9 = polygeist.submap(%alloca_5, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          %10 = polygeist.submap(%alloca_6, %c5, %c4) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "reduction"]} ins(%3, %4, %5, %6, %7 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%8, %9, %10 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) {
          ^bb0(%in: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %out: f64, %out_11: f64, %out_12: f64):
            %17 = arith.mulf %in, %in_7 : f64
            %18 = arith.addf %out_12, %17 : f64
            %19 = arith.mulf %in_8, %in_7 : f64
            %20 = arith.addf %out_11, %19 : f64
            %21 = arith.mulf %in_9, %in_10 : f64
            %22 = arith.addf %out, %21 : f64
            linalg.yield %22, %20, %18 : f64, f64, f64
          }
          %11 = polygeist.submap(%alloca_6, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %12 = polygeist.submap(%arg3, %arg4, %arg5, %arg6, %c5) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%11 : memref<?xf64>) outs(%12 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %13 = polygeist.submap(%alloca_5, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %14 = polygeist.submap(%arg3, %arg4, %arg5, %arg6, %c5) {map = #map10} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : memref<?xf64>) outs(%14 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
          %15 = polygeist.submap(%alloca_4, %c5) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %16 = polygeist.submap(%arg3, %arg4, %arg5, %arg6, %c5) {map = #map11} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%15 : memref<?xf64>) outs(%16 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            linalg.yield %in : f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    }
    return
  }
}
