#map = affine_map<(d0)[s0] -> (d0 + s0 * 4)>
#map1 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 64 + s1 * 16 + s2 * 4)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0)[s0] -> (d0 + s0 * 5)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map6 = affine_map<(d0, d1, d2, d3)[s0] -> (d2 + s0 * 64 + d0 * 16 + d1 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
          affine.for %arg9 = 0 to 5 {
            affine.store %cst, %alloca_8[%arg7, %arg8, %arg9] : memref<4x4x5xf64>
            affine.store %cst, %alloca_9[%arg7, %arg8, %arg9] : memref<4x4x5xf64>
            %alloca_10 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_10[] : memref<f64>
            %alloca_11 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_11[] : memref<f64>
            %2 = polygeist.submap(%arg0, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %3 = polygeist.submap(%arg4, %arg6, %arg7, %arg8, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
            %4 = polygeist.submap(%arg1, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview = memref.subview %alloca_9[%arg7, %arg8, %arg9] [1, 1, 1] [1, 1, 1] : memref<4x4x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_12 = memref.subview %alloca_8[%arg7, %arg8, %arg9] [1, 1, 1] [1, 1, 1] : memref<4x4x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_13 = memref.subview %alloca_10[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_14 = memref.subview %alloca_11[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map2, #map2, #map2, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %3, %4 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%subview, %subview_12, %subview_13, %subview_14 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64, %out_18: f64, %out_19: f64):
              %5 = arith.mulf %in, %in_15 : f64
              %6 = arith.addf %out_19, %5 : f64
              %7 = arith.mulf %in_16, %in_15 : f64
              %8 = arith.addf %out_18, %7 : f64
              linalg.yield %6, %8, %8, %6 : f64, f64, f64, f64
            }
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 5 {
            affine.store %cst, %alloca_5[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
            affine.store %cst, %alloca_6[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
            affine.store %cst, %alloca_7[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
            %alloca_10 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_10[] : memref<f64>
            %alloca_11 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_11[] : memref<f64>
            %alloca_12 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_12[] : memref<f64>
            %2 = polygeist.submap(%arg0, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview = memref.subview %alloca_9[%arg7, 0, %arg8] [1, %c4, 1] [1, 1, 1] : memref<4x4x5xf64> to memref<?xf64, strided<[5], offset: ?>>
            %3 = polygeist.submap(%arg1, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %4 = polygeist.submap(%arg0, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_13 = memref.subview %alloca_8[%arg7, 0, %arg8] [1, %c4, 1] [1, 1, 1] : memref<4x4x5xf64> to memref<?xf64, strided<[5], offset: ?>>
            %subview_14 = memref.subview %alloca_7[%arg7, %arg9, %arg8] [1, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_15 = memref.subview %alloca_6[%arg7, %arg9, %arg8] [1, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_16 = memref.subview %alloca_5[%arg7, %arg9, %arg8] [1, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_17 = memref.subview %alloca_10[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_18 = memref.subview %alloca_11[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_19 = memref.subview %alloca_12[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2, #map3, #map3, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview, %3, %4, %subview_13 : memref<?xf64>, memref<?xf64, strided<[5], offset: ?>>, memref<?xf64>, memref<?xf64>, memref<?xf64, strided<[5], offset: ?>>) outs(%subview_14, %subview_15, %subview_16, %subview_17, %subview_18, %subview_19 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_20: f64, %in_21: f64, %in_22: f64, %in_23: f64, %out: f64, %out_24: f64, %out_25: f64, %out_26: f64, %out_27: f64, %out_28: f64):
              %5 = arith.mulf %in, %in_20 : f64
              %6 = arith.addf %out_28, %5 : f64
              %7 = arith.mulf %in_21, %in_20 : f64
              %8 = arith.addf %out_27, %7 : f64
              %9 = arith.mulf %in_22, %in_23 : f64
              %10 = arith.addf %out_26, %9 : f64
              linalg.yield %6, %8, %10, %10, %8, %6 : f64, f64, f64, f64, f64, f64
            }
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 5 {
            affine.store %cst, %alloca_2[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            affine.store %cst, %alloca_3[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            affine.store %cst, %alloca_4[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %alloca_10 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_10[] : memref<f64>
            %alloca_11 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_11[] : memref<f64>
            %alloca_12 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_12[] : memref<f64>
            %2 = polygeist.submap(%arg1, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview = memref.subview %alloca_7[0, %arg8, %arg7] [%c4, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<?xf64, strided<[25], offset: ?>>
            %3 = polygeist.submap(%arg0, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_13 = memref.subview %alloca_6[0, %arg8, %arg7] [%c4, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<?xf64, strided<[25], offset: ?>>
            %4 = polygeist.submap(%arg0, %arg9, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_14 = memref.subview %alloca_5[0, %arg8, %arg7] [%c4, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<?xf64, strided<[25], offset: ?>>
            %subview_15 = memref.subview %alloca_4[%arg9, %arg8, %arg7] [1, 1, 1] [1, 1, 1] : memref<5x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_16 = memref.subview %alloca_3[%arg9, %arg8, %arg7] [1, 1, 1] [1, 1, 1] : memref<5x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_17 = memref.subview %alloca_2[%arg9, %arg8, %arg7] [1, 1, 1] [1, 1, 1] : memref<5x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_18 = memref.subview %alloca_10[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_19 = memref.subview %alloca_11[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_20 = memref.subview %alloca_12[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2, #map2, #map3, #map3, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview, %3, %subview_13, %4, %subview_14 : memref<?xf64>, memref<?xf64, strided<[25], offset: ?>>, memref<?xf64>, memref<?xf64, strided<[25], offset: ?>>, memref<?xf64>, memref<?xf64, strided<[25], offset: ?>>) outs(%subview_15, %subview_16, %subview_17, %subview_18, %subview_19, %subview_20 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_21: f64, %in_22: f64, %in_23: f64, %in_24: f64, %in_25: f64, %out: f64, %out_26: f64, %out_27: f64, %out_28: f64, %out_29: f64, %out_30: f64):
              %16 = arith.mulf %in, %in_21 : f64
              %17 = arith.addf %out_30, %16 : f64
              %18 = arith.mulf %in_22, %in_23 : f64
              %19 = arith.addf %out_29, %18 : f64
              %20 = arith.mulf %in_24, %in_25 : f64
              %21 = arith.addf %out_28, %20 : f64
              linalg.yield %17, %19, %21, %21, %19, %17 : f64, f64, f64, f64, f64, f64
            }
            %5 = affine.load %arg3[%arg6 * 375 + %arg7 + %arg9 * 25 + %arg8 * 5] : memref<?xf64>
            %6 = affine.load %alloca_2[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %7 = arith.mulf %5, %6 : f64
            %8 = affine.load %arg3[%arg6 * 375 + %arg7 + %arg9 * 25 + %arg8 * 5 + 125] : memref<?xf64>
            %9 = affine.load %alloca_3[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %10 = arith.mulf %8, %9 : f64
            %11 = arith.addf %7, %10 : f64
            %12 = affine.load %arg3[%arg6 * 375 + %arg7 + %arg9 * 25 + %arg8 * 5 + 250] : memref<?xf64>
            %13 = affine.load %alloca_4[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %14 = arith.mulf %12, %13 : f64
            %15 = arith.addf %11, %14 : f64
            affine.store %15, %alloca_1[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 4 {
            affine.store %cst, %alloca_0[%arg9, %arg8, %arg7] : memref<4x5x5xf64>
            %alloca_10 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_10[] : memref<f64>
            %2 = polygeist.submap(%arg2, %arg9, %c5) {map = #map4} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview = memref.subview %alloca_1[0, %arg8, %arg7] [%c5, 1, 1] [1, 1, 1] : memref<5x5x5xf64> to memref<?xf64, strided<[25], offset: ?>>
            %subview_11 = memref.subview %alloca_0[%arg9, %arg8, %arg7] [1, 1, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_12 = memref.subview %alloca_10[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map2, #map2, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview : memref<?xf64>, memref<?xf64, strided<[25], offset: ?>>) outs(%subview_11, %subview_12 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_13: f64, %out: f64, %out_14: f64):
              %3 = arith.mulf %in, %in_13 : f64
              %4 = arith.addf %out_14, %3 : f64
              linalg.yield %4, %4 : f64, f64
            }
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 4 {
            affine.store %cst, %alloca[%arg7, %arg9, %arg8] : memref<4x4x5xf64>
            %alloca_10 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_10[] : memref<f64>
            %2 = polygeist.submap(%arg2, %arg9, %c5) {map = #map4} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview = memref.subview %alloca_0[%arg7, 0, %arg8] [1, %c5, 1] [1, 1, 1] : memref<4x5x5xf64> to memref<?xf64, strided<[5], offset: ?>>
            %subview_11 = memref.subview %alloca[%arg7, %arg9, %arg8] [1, 1, 1] [1, 1, 1] : memref<4x4x5xf64> to memref<f64, strided<[], offset: ?>>
            %subview_12 = memref.subview %alloca_10[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map2, #map2, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview : memref<?xf64>, memref<?xf64, strided<[5], offset: ?>>) outs(%subview_11, %subview_12 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_13: f64, %out: f64, %out_14: f64):
              %3 = arith.mulf %in, %in_13 : f64
              %4 = arith.addf %out_14, %3 : f64
              linalg.yield %4, %4 : f64, f64
            }
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      %0 = polygeist.submap(%arg2, %c4, %c4, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
      %1 = polygeist.submap(%arg5, %arg6, %c4, %c4, %c4, %c5) {map = #map6} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?x?xf64>
      linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %alloca : memref<?x?x?x?xf64>, memref<4x4x5xf64>) outs(%1 : memref<?x?x?x?xf64>) {
      ^bb0(%in: f64, %in_10: f64, %out: f64):
        %2 = arith.mulf %in, %in_10 : f64
        %3 = arith.addf %out, %2 : f64
        linalg.yield %3 : f64
      }
    }
    return
  }
}
