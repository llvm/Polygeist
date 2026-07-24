#map = affine_map<(d0)[s0] -> (d0 + s0 * 4)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 + s0 * 16 + s1 * 4)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0)[s0] -> (d0 + s0 * 5)>
#map5 = affine_map<(d0, d1, d2) -> (d2 + d0 * 5)>
#map6 = affine_map<(d0, d1, d2)[s0] -> (d1 * 4 + d0 + s0 * 16)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x5xf64>
    %alloca_0 = memref.alloca() : memref<5x5xf64>
    %alloca_1 = memref.alloca() : memref<5x5xf64>
    %alloca_2 = memref.alloca() : memref<5x5xf64>
    %alloca_3 = memref.alloca() : memref<4x5xf64>
    %alloca_4 = memref.alloca() : memref<4x5xf64>
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          affine.store %cst, %alloca_3[%arg7, %arg8] : memref<4x5xf64>
          affine.store %cst, %alloca_4[%arg7, %arg8] : memref<4x5xf64>
          %alloca_5 = memref.alloca() : memref<f64>
          affine.store %cst, %alloca_5[] : memref<f64>
          %alloca_6 = memref.alloca() : memref<f64>
          affine.store %cst, %alloca_6[] : memref<f64>
          %2 = polygeist.submap(%arg0, %arg8, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
          %3 = polygeist.submap(%arg4, %arg6, %arg7, %c4) {map = #map1} : (memref<?xf64>, index, index, index) -> memref<?xf64>
          %4 = polygeist.submap(%arg1, %arg8, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview = memref.subview %alloca_4[%arg7, %arg8] [1, 1] [1, 1] : memref<4x5xf64> to memref<f64, strided<[], offset: ?>>
          %subview_7 = memref.subview %alloca_3[%arg7, %arg8] [1, 1] [1, 1] : memref<4x5xf64> to memref<f64, strided<[], offset: ?>>
          %subview_8 = memref.subview %alloca_5[] [] [] : memref<f64> to memref<f64, strided<[]>>
          %subview_9 = memref.subview %alloca_6[] [] [] : memref<f64> to memref<f64, strided<[]>>
          linalg.generic {indexing_maps = [#map2, #map2, #map2, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %3, %4 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%subview, %subview_7, %subview_8, %subview_9 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
          ^bb0(%in: f64, %in_10: f64, %in_11: f64, %out: f64, %out_12: f64, %out_13: f64, %out_14: f64):
            %5 = arith.mulf %in, %in_10 : f64
            %6 = arith.addf %out_14, %5 : f64
            %7 = arith.mulf %in_11, %in_10 : f64
            %8 = arith.addf %out_13, %7 : f64
            linalg.yield %6, %8, %8, %6 : f64, f64, f64, f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          affine.store %cst, %alloca_1[%arg8, %arg7] : memref<5x5xf64>
          affine.store %cst, %alloca_2[%arg8, %arg7] : memref<5x5xf64>
          %alloca_5 = memref.alloca() : memref<f64>
          affine.store %cst, %alloca_5[] : memref<f64>
          %alloca_6 = memref.alloca() : memref<f64>
          affine.store %cst, %alloca_6[] : memref<f64>
          %2 = polygeist.submap(%arg1, %arg8, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview = memref.subview %alloca_4[0, %arg7] [%c4, 1] [1, 1] : memref<4x5xf64> to memref<?xf64, strided<[5], offset: ?>>
          %3 = polygeist.submap(%arg0, %arg8, %c4) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview_7 = memref.subview %alloca_3[0, %arg7] [%c4, 1] [1, 1] : memref<4x5xf64> to memref<?xf64, strided<[5], offset: ?>>
          %subview_8 = memref.subview %alloca_2[%arg8, %arg7] [1, 1] [1, 1] : memref<5x5xf64> to memref<f64, strided<[], offset: ?>>
          %subview_9 = memref.subview %alloca_1[%arg8, %arg7] [1, 1] [1, 1] : memref<5x5xf64> to memref<f64, strided<[], offset: ?>>
          %subview_10 = memref.subview %alloca_5[] [] [] : memref<f64> to memref<f64, strided<[]>>
          %subview_11 = memref.subview %alloca_6[] [] [] : memref<f64> to memref<f64, strided<[]>>
          linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview, %3, %subview_7 : memref<?xf64>, memref<?xf64, strided<[5], offset: ?>>, memref<?xf64>, memref<?xf64, strided<[5], offset: ?>>) outs(%subview_8, %subview_9, %subview_10, %subview_11 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
          ^bb0(%in: f64, %in_12: f64, %in_13: f64, %in_14: f64, %out: f64, %out_15: f64, %out_16: f64, %out_17: f64):
            %11 = arith.mulf %in, %in_12 : f64
            %12 = arith.addf %out_17, %11 : f64
            %13 = arith.mulf %in_13, %in_14 : f64
            %14 = arith.addf %out_16, %13 : f64
            linalg.yield %12, %14, %14, %12 : f64, f64, f64, f64
          }
          %4 = affine.load %arg3[%arg7 + %arg6 * 50 + %arg8 * 5] : memref<?xf64>
          %5 = affine.load %alloca_1[%arg8, %arg7] : memref<5x5xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %arg3[%arg7 + %arg6 * 50 + %arg8 * 5 + 25] : memref<?xf64>
          %8 = affine.load %alloca_2[%arg8, %arg7] : memref<5x5xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %6, %9 : f64
          affine.store %10, %alloca_0[%arg8, %arg7] : memref<5x5xf64>
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 4 {
          affine.store %cst, %alloca[%arg8, %arg7] : memref<4x5xf64>
          %alloca_5 = memref.alloca() : memref<f64>
          affine.store %cst, %alloca_5[] : memref<f64>
          %2 = polygeist.submap(%arg2, %arg8, %c5) {map = #map4} : (memref<?xf64>, index, index) -> memref<?xf64>
          %subview = memref.subview %alloca_0[0, %arg7] [%c5, 1] [1, 1] : memref<5x5xf64> to memref<?xf64, strided<[5], offset: ?>>
          %subview_6 = memref.subview %alloca[%arg8, %arg7] [1, 1] [1, 1] : memref<4x5xf64> to memref<f64, strided<[], offset: ?>>
          %subview_7 = memref.subview %alloca_5[] [] [] : memref<f64> to memref<f64, strided<[]>>
          linalg.generic {indexing_maps = [#map2, #map2, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview : memref<?xf64>, memref<?xf64, strided<[5], offset: ?>>) outs(%subview_6, %subview_7 : memref<f64, strided<[], offset: ?>>, memref<f64, strided<[]>>) {
          ^bb0(%in: f64, %in_8: f64, %out: f64, %out_9: f64):
            %3 = arith.mulf %in, %in_8 : f64
            %4 = arith.addf %out_9, %3 : f64
            linalg.yield %4, %4 : f64, f64
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      %0 = polygeist.submap(%arg2, %c4, %c4, %c5) {map = #map5} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
      %1 = polygeist.submap(%arg5, %arg6, %c4, %c4, %c5) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %alloca : memref<?x?x?xf64>, memref<4x5xf64>) outs(%1 : memref<?x?x?xf64>) {
      ^bb0(%in: f64, %in_5: f64, %out: f64):
        %2 = arith.mulf %in, %in_5 : f64
        %3 = arith.addf %out, %2 : f64
        linalg.yield %3 : f64
      }
    }
    return
  }
}
