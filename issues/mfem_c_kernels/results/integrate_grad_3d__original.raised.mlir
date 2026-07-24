#map = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5)>
#map1 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map2 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5 + 125)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (d0 * 25 + s0 * 375 + s1 + s2 * 5 + 250)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_grad_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
          affine.for %arg7 = 0 to 4 {
            %alloca_5 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_5[] : memref<f64>
            %alloca_6 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_6[] : memref<f64>
            %alloca_7 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_7[] : memref<f64>
            %0 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c5) {map = #map} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
            %1 = polygeist.submap(%arg2, %arg7, %c5) {map = #map1} : (memref<?xf64>, index, index) -> memref<?xf64>
            %2 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c5) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
            %3 = polygeist.submap(%arg1, %arg7, %c5) {map = #map1} : (memref<?xf64>, index, index) -> memref<?xf64>
            %4 = polygeist.submap(%arg0, %arg4, %arg5, %arg6, %c5) {map = #map3} : (memref<?xf64>, index, index, index, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map5, #map5, #map5], iterator_types = ["reduction"]} ins(%0, %1, %2, %3, %4 : memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%alloca_5, %alloca_6, %alloca_7 : memref<f64>, memref<f64>, memref<f64>) {
            ^bb0(%in: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %out: f64, %out_12: f64, %out_13: f64):
              %8 = arith.mulf %in, %in_8 : f64
              %9 = arith.addf %out_13, %8 : f64
              %10 = arith.mulf %in_9, %in_10 : f64
              %11 = arith.addf %out_12, %10 : f64
              %12 = arith.mulf %in_11, %in_10 : f64
              %13 = arith.addf %out, %12 : f64
              linalg.yield %13, %11, %9 : f64, f64, f64
            }
            %5 = affine.load %alloca_5[] : memref<f64>
            %6 = affine.load %alloca_6[] : memref<f64>
            %7 = affine.load %alloca_7[] : memref<f64>
            affine.store %7, %alloca_4[%arg5, %arg6, %arg7] : memref<5x5x4xf64>
            affine.store %6, %alloca_3[%arg5, %arg6, %arg7] : memref<5x5x4xf64>
            affine.store %5, %alloca_2[%arg5, %arg6, %arg7] : memref<5x5x4xf64>
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %alloca_5 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_5[] : memref<f64>
            %alloca_6 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_6[] : memref<f64>
            %alloca_7 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_7[] : memref<f64>
            %subview = memref.subview %alloca_4[%arg5, 0, %arg7] [1, %c5, 1] [1, 1, 1] : memref<5x5x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            %0 = polygeist.submap(%arg1, %arg6, %c5) {map = #map1} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_8 = memref.subview %alloca_3[%arg5, 0, %arg7] [1, %c5, 1] [1, 1, 1] : memref<5x5x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            %1 = polygeist.submap(%arg2, %arg6, %c5) {map = #map1} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_9 = memref.subview %alloca_2[%arg5, 0, %arg7] [1, %c5, 1] [1, 1, 1] : memref<5x5x4xf64> to memref<?xf64, strided<[4], offset: ?>>
            %subview_10 = memref.subview %alloca_5[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_11 = memref.subview %alloca_6[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_12 = memref.subview %alloca_7[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map5, #map5, #map5], iterator_types = ["reduction"]} ins(%subview, %0, %subview_8, %1, %subview_9 : memref<?xf64, strided<[4], offset: ?>>, memref<?xf64>, memref<?xf64, strided<[4], offset: ?>>, memref<?xf64>, memref<?xf64, strided<[4], offset: ?>>) outs(%subview_10, %subview_11, %subview_12 : memref<f64, strided<[]>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64, %out_18: f64):
              %5 = arith.mulf %in, %in_13 : f64
              %6 = arith.addf %out_18, %5 : f64
              %7 = arith.mulf %in_14, %in_15 : f64
              %8 = arith.addf %out_17, %7 : f64
              %9 = arith.mulf %in_16, %in_13 : f64
              %10 = arith.addf %out, %9 : f64
              linalg.yield %10, %8, %6 : f64, f64, f64
            }
            %2 = affine.load %alloca_5[] : memref<f64>
            %3 = affine.load %alloca_6[] : memref<f64>
            %4 = affine.load %alloca_7[] : memref<f64>
            affine.store %4, %alloca_1[%arg5, %arg6, %arg7] : memref<5x4x4xf64>
            affine.store %3, %alloca_0[%arg5, %arg6, %arg7] : memref<5x4x4xf64>
            affine.store %2, %alloca[%arg5, %arg6, %arg7] : memref<5x4x4xf64>
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %alloca_5 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_5[] : memref<f64>
            %alloca_6 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_6[] : memref<f64>
            %alloca_7 = memref.alloca() : memref<f64>
            affine.store %cst, %alloca_7[] : memref<f64>
            %subview = memref.subview %alloca_1[0, %arg6, %arg7] [%c5, 1, 1] [1, 1, 1] : memref<5x4x4xf64> to memref<?xf64, strided<[16], offset: ?>>
            %0 = polygeist.submap(%arg1, %arg5, %c5) {map = #map1} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_8 = memref.subview %alloca_0[0, %arg6, %arg7] [%c5, 1, 1] [1, 1, 1] : memref<5x4x4xf64> to memref<?xf64, strided<[16], offset: ?>>
            %subview_9 = memref.subview %alloca[0, %arg6, %arg7] [%c5, 1, 1] [1, 1, 1] : memref<5x4x4xf64> to memref<?xf64, strided<[16], offset: ?>>
            %1 = polygeist.submap(%arg2, %arg5, %c5) {map = #map1} : (memref<?xf64>, index, index) -> memref<?xf64>
            %subview_10 = memref.subview %alloca_5[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_11 = memref.subview %alloca_6[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_12 = memref.subview %alloca_7[] [] [] : memref<f64> to memref<f64, strided<[]>>
            linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map5, #map5, #map5], iterator_types = ["reduction"]} ins(%subview, %0, %subview_8, %subview_9, %1 : memref<?xf64, strided<[16], offset: ?>>, memref<?xf64>, memref<?xf64, strided<[16], offset: ?>>, memref<?xf64, strided<[16], offset: ?>>, memref<?xf64>) outs(%subview_10, %subview_11, %subview_12 : memref<f64, strided<[]>>, memref<f64, strided<[]>>, memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %out: f64, %out_17: f64, %out_18: f64):
              %9 = arith.mulf %in, %in_13 : f64
              %10 = arith.addf %out_18, %9 : f64
              %11 = arith.mulf %in_14, %in_13 : f64
              %12 = arith.addf %out_17, %11 : f64
              %13 = arith.mulf %in_15, %in_16 : f64
              %14 = arith.addf %out, %13 : f64
              linalg.yield %14, %12, %10 : f64, f64, f64
            }
            %2 = affine.load %alloca_5[] : memref<f64>
            %3 = affine.load %alloca_6[] : memref<f64>
            %4 = affine.load %alloca_7[] : memref<f64>
            %5 = arith.addf %4, %3 : f64
            %6 = arith.addf %5, %2 : f64
            %7 = affine.load %arg3[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
            %8 = arith.addf %7, %6 : f64
            affine.store %8, %arg3[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    }
    return
  }
}
