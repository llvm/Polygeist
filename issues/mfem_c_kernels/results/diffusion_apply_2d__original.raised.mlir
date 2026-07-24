#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0, d1)[s0] -> (d1 + d0 * 5 + s0 * 75)>
#map4 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 75 + d0 * 5 + 25)>
#map5 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 75 + d0 * 5 + 50)>
#map6 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map7 = affine_map<(d0, d1)[s0] -> (d0 * 5 + s0)>
#map8 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 16 + d0 * 4)>
#map9 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x2xf64>
    %alloca_0 = memref.alloca() : memref<5x2xf64>
    %alloca_1 = memref.alloca() : memref<5x5x2xf64>
    affine.for %arg7 = 0 to 2 {
      %subview = memref.subview %alloca_1[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf64, strided<[10, 2], offset: 1>>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %subview_2 = memref.subview %alloca_1[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview_2 : memref<?x?xf64, strided<[10, 2]>>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg8 = 0 to 4 {
        %subview_5 = memref.subview %alloca_0[0, 1] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2], offset: 1>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_5 : memref<?xf64, strided<[2], offset: 1>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %subview_6 = memref.subview %alloca_0[0, 0] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2]>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_6 : memref<?xf64, strided<[2]>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 4 {
          %3 = affine.load %arg5[%arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
          %4 = polygeist.submap(%arg0, %arg9, %c5) {map = #map2} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_7 = memref.subview %alloca_0[0, 0] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2]>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : memref<?xf64>) outs(%subview_7 : memref<?xf64, strided<[2]>>) {
          ^bb0(%in: f64, %out: f64):
            %6 = arith.mulf %3, %in : f64
            %7 = arith.addf %out, %6 : f64
            linalg.yield %7 : f64
          }
          %5 = polygeist.submap(%arg1, %arg9, %c5) {map = #map2} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_8 = memref.subview %alloca_0[0, 1] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%5 : memref<?xf64>) outs(%subview_8 : memref<?xf64, strided<[2], offset: 1>>) {
          ^bb0(%in: f64, %out: f64):
            %6 = arith.mulf %3, %in : f64
            %7 = arith.addf %out, %6 : f64
            linalg.yield %7 : f64
          }
        }
        affine.for %arg9 = 0 to 5 {
          %3 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
          %4 = affine.load %arg1[%arg8 + %arg9 * 4] : memref<?xf64>
          %subview_7 = memref.subview %alloca_0[0, 1] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          %subview_8 = memref.subview %alloca_1[%arg9, 0, 0] [1, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?xf64, strided<[2], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%subview_7 : memref<?xf64, strided<[2], offset: 1>>) outs(%subview_8 : memref<?xf64, strided<[2], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %5 = arith.mulf %in, %3 : f64
            %6 = arith.addf %out, %5 : f64
            linalg.yield %6 : f64
          }
          %subview_9 = memref.subview %alloca_0[0, 0] [%c5, 1] [1, 1] : memref<5x2xf64> to memref<?xf64, strided<[2]>>
          %subview_10 = memref.subview %alloca_1[%arg9, 0, 1] [1, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?xf64, strided<[2], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%subview_9 : memref<?xf64, strided<[2]>>) outs(%subview_10 : memref<?xf64, strided<[2], offset: ?>>) {
          ^bb0(%in: f64, %out: f64):
            %5 = arith.mulf %in, %4 : f64
            %6 = arith.addf %out, %5 : f64
            linalg.yield %6 : f64
          }
        } {polygeist.was_parallel}
      }
      %0 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map3} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %1 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map4} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %2 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map5} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %subview_3 = memref.subview %alloca_1[0, 0, 0] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2]>>
      %subview_4 = memref.subview %alloca_1[0, 0, 1] [%c5, %c5, 1] [1, 1, 1] : memref<5x5x2xf64> to memref<?x?xf64, strided<[10, 2], offset: 1>>
      linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %2 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%subview_3, %subview_4 : memref<?x?xf64, strided<[10, 2]>>, memref<?x?xf64, strided<[10, 2], offset: 1>>) {
      ^bb0(%in: f64, %in_5: f64, %in_6: f64, %out: f64, %out_7: f64):
        %3 = arith.mulf %in, %out : f64
        %4 = arith.mulf %in_5, %out_7 : f64
        %5 = arith.addf %3, %4 : f64
        %6 = arith.mulf %in_5, %out : f64
        %7 = arith.mulf %in_6, %out_7 : f64
        %8 = arith.addf %6, %7 : f64
        linalg.yield %5, %8 : f64, f64
      }
      affine.for %arg8 = 0 to 5 {
        %subview_5 = memref.subview %alloca[0, 1] [%c4, 1] [1, 1] : memref<4x2xf64> to memref<?xf64, strided<[2], offset: 1>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_5 : memref<?xf64, strided<[2], offset: 1>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %subview_6 = memref.subview %alloca[0, 0] [%c4, 1] [1, 1] : memref<4x2xf64> to memref<?xf64, strided<[2]>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_6 : memref<?xf64, strided<[2]>>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 5 {
          %6 = affine.load %alloca_1[%arg8, %arg9, 0] : memref<5x5x2xf64>
          %7 = affine.load %alloca_1[%arg8, %arg9, 1] : memref<5x5x2xf64>
          %8 = polygeist.submap(%arg3, %arg9, %c4) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_9 = memref.subview %alloca[0, 0] [%c4, 1] [1, 1] : memref<4x2xf64> to memref<?xf64, strided<[2]>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%8 : memref<?xf64>) outs(%subview_9 : memref<?xf64, strided<[2]>>) {
          ^bb0(%in: f64, %out: f64):
            %10 = arith.mulf %6, %in : f64
            %11 = arith.addf %out, %10 : f64
            linalg.yield %11 : f64
          }
          %9 = polygeist.submap(%arg2, %arg9, %c4) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_10 = memref.subview %alloca[0, 1] [%c4, 1] [1, 1] : memref<4x2xf64> to memref<?xf64, strided<[2], offset: 1>>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%9 : memref<?xf64>) outs(%subview_10 : memref<?xf64, strided<[2], offset: 1>>) {
          ^bb0(%in: f64, %out: f64):
            %10 = arith.mulf %7, %in : f64
            %11 = arith.addf %out, %10 : f64
            linalg.yield %11 : f64
          }
        }
        %subview_7 = memref.subview %alloca[0, 0] [%c4, 1] [1, 1] : memref<4x2xf64> to memref<?xf64, strided<[2]>>
        %3 = polygeist.submap(%arg2, %arg8, %c4, %c4) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        %subview_8 = memref.subview %alloca[0, 1] [%c4, 1] [1, 1] : memref<4x2xf64> to memref<?xf64, strided<[2], offset: 1>>
        %4 = polygeist.submap(%arg3, %arg8, %c4, %c4) {map = #map7} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        %5 = polygeist.submap(%arg6, %arg7, %c4, %c4) {map = #map8} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map9, #map, #map9, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_7, %3, %subview_8, %4 : memref<?xf64, strided<[2]>>, memref<?x?xf64>, memref<?xf64, strided<[2], offset: 1>>, memref<?x?xf64>) outs(%5 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_9: f64, %in_10: f64, %in_11: f64, %out: f64):
          %6 = arith.mulf %in, %in_9 : f64
          %7 = arith.mulf %in_10, %in_11 : f64
          %8 = arith.addf %6, %7 : f64
          %9 = arith.addf %out, %8 : f64
          linalg.yield %9 : f64
        }
      }
    }
    return
  }
}
