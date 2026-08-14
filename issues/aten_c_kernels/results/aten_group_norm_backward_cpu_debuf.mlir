#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_group_norm_backward_cpu(%arg0: memref<?x4x2x16xf32>, %arg1: memref<?x4x2x16xf32>, %arg2: memref<?x4xf32>, %arg3: memref<?x4xf32>, %arg4: memref<?x2xf32>, %arg5: memref<?x4x2x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg6 = 0 to 4 {
      %alloca = memref.alloca(%c4) : memref<?xf32>
      %alloca_1 = memref.alloca(%c4) : memref<?xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_1 : memref<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      affine.for %arg7 = 0 to 4 {
        %2 = affine.load %arg2[%arg6, %arg7] : memref<?x4xf32>
        %alloca_9 = memref.alloca(%c2) : memref<?xf32>
        %alloca_10 = memref.alloca(%c2) : memref<?xf32>
        affine.for %arg8 = 0 to 2 {
          %3 = affine.load %alloca[%arg7] : memref<?xf32>
          %4 = affine.load %alloca_1[%arg7] : memref<?xf32>
          %5 = affine.load %arg4[%arg7, %arg8] : memref<?x2xf32>
          affine.store %3, %alloca_9[%arg8] : memref<?xf32>
          affine.store %4, %alloca_10[%arg8] : memref<?xf32>
          %subview_11 = memref.subview %arg0[%arg6, %arg7, %arg8, 0] [1, 1, 1, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_12 = memref.subview %arg1[%arg6, %arg7, %arg8, 0] [1, 1, 1, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_13 = memref.subview %alloca_9[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          %subview_14 = memref.subview %alloca_10[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["reduction"]} ins(%subview_11, %subview_12 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_13, %subview_14 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %in_15: f32, %out: f32, %out_16: f32):
            %8 = arith.mulf %in, %5 : f32
            %9 = arith.addf %out_16, %8 : f32
            %10 = arith.subf %in_15, %2 : f32
            %11 = arith.mulf %8, %10 : f32
            %12 = arith.addf %out, %11 : f32
            linalg.yield %12, %9 : f32, f32
          }
          %6 = affine.load %alloca_9[%arg8] : memref<?xf32>
          %7 = affine.load %alloca_10[%arg8] : memref<?xf32>
          affine.store %6, %alloca[%arg7] : memref<?xf32>
          affine.store %7, %alloca_1[%arg7] : memref<?xf32>
        }
      } {polygeist.was_parallel}
      %subview = memref.subview %arg0[%arg6, 0, 0, 0] [1, %c4, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?x?xf32, strided<[32, 16, 1], offset: ?>>
      %subview_2 = memref.subview %arg4[0, 0] [%c4, %c2] [1, 1] : memref<?x2xf32> to memref<?x?xf32, strided<[2, 1]>>
      %subview_3 = memref.subview %arg3[%arg6, 0] [1, %c4] [1, 1] : memref<?x4xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %arg1[%arg6, 0, 0, 0] [1, %c4, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?x?xf32, strided<[32, 16, 1], offset: ?>>
      %subview_5 = memref.subview %arg2[%arg6, 0] [1, %c4] [1, 1] : memref<?x4xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg5[%arg6, 0, 0, 0] [1, %c4, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?x?xf32, strided<[32, 16, 1], offset: ?>>
      %0 = polygeist.submap(%alloca, %c4) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
      %subview_7 = memref.subview %0[0] [%c4] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %1 = polygeist.submap(%alloca_1, %c4) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
      %subview_8 = memref.subview %1[0] [%c4] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map2, #map2, #map3, #map4, #map2, #map3, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_7, %subview_8, %subview, %subview_2, %subview_3, %subview_4, %subview_5 : memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>, memref<?x?x?xf32, strided<[32, 16, 1], offset: ?>>, memref<?x?xf32, strided<[2, 1]>>, memref<?xf32, strided<[1], offset: ?>>, memref<?x?x?xf32, strided<[32, 16, 1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_6 : memref<?x?x?xf32, strided<[32, 16, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %in_12: f32, %in_13: f32, %in_14: f32, %out: f32):
        %2 = arith.mulf %in_10, %in_11 : f32
        %3 = arith.divf %in_12, %cst : f32
        %4 = arith.mulf %2, %cst : f32
        %5 = arith.subf %4, %in_9 : f32
        %6 = arith.subf %in_13, %in_14 : f32
        %7 = arith.mulf %6, %in_12 : f32
        %8 = arith.mulf %7, %in_12 : f32
        %9 = arith.mulf %8, %in : f32
        %10 = arith.subf %5, %9 : f32
        %11 = arith.mulf %3, %10 : f32
        linalg.yield %11 : f32
      }
    } {polygeist.was_parallel}
    return
  }
}

