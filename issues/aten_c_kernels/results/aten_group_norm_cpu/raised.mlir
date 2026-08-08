#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_group_norm_cpu(%arg0: memref<?x4x2x16xf32>, %arg1: memref<?x2xf32>, %arg2: memref<?x2xf32>, %arg3: f32, %arg4: memref<?x4x2x16xf32>, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    affine.for %arg7 = 0 to 4 {
      affine.for %arg8 = 0 to 4 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_1, %alloca[] : memref<f32>
        %subview = memref.subview %arg0[%arg7, %arg8, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%subview : memref<?x?xf32, strided<[16, 1], offset: ?>>) outs(%alloca : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %out, %in : f32
          linalg.yield %7 : f32
        }
        %0 = affine.load %alloca[] : memref<f32>
        %1 = arith.divf %0, %cst : f32
        affine.store %1, %arg5[%arg7, %arg8] : memref<?x4xf32>
        %alloca_2 = memref.alloca() : memref<f32>
        affine.store %cst_1, %alloca_2[] : memref<f32>
        %subview_3 = memref.subview %arg0[%arg7, %arg8, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%subview_3 : memref<?x?xf32, strided<[16, 1], offset: ?>>) outs(%alloca_2 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %1 : f32
          %8 = arith.mulf %7, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        }
        %2 = affine.load %alloca_2[] : memref<f32>
        %3 = arith.divf %2, %cst : f32
        %4 = arith.addf %3, %arg3 : f32
        %5 = math.sqrt %4 : f32
        %6 = arith.divf %cst_0, %5 : f32
        affine.store %6, %arg6[%arg7, %arg8] : memref<?x4xf32>
        %subview_4 = memref.subview %arg0[%arg7, %arg8, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
        %subview_5 = memref.subview %arg5[%arg7, %arg8] [1, 1] [1, 1] : memref<?x4xf32> to memref<f32, strided<[], offset: ?>>
        %subview_6 = memref.subview %arg6[%arg7, %arg8] [1, 1] [1, 1] : memref<?x4xf32> to memref<f32, strided<[], offset: ?>>
        %subview_7 = memref.subview %arg1[%arg8, 0] [1, %c2] [1, 1] : memref<?x2xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_8 = memref.subview %arg2[%arg8, 0] [1, %c2] [1, 1] : memref<?x2xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_9 = memref.subview %arg4[%arg7, %arg8, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : memref<?x4x2x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map1, #map1, #map2, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_4, %subview_5, %subview_6, %subview_7, %subview_8 : memref<?x?xf32, strided<[16, 1], offset: ?>>, memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_9 : memref<?x?xf32, strided<[16, 1], offset: ?>>) {
        ^bb0(%in: f32, %in_10: f32, %in_11: f32, %in_12: f32, %in_13: f32, %out: f32):
          %7 = arith.subf %in, %in_10 : f32
          %8 = arith.mulf %7, %in_11 : f32
          %9 = arith.mulf %8, %in_12 : f32
          %10 = arith.addf %9, %in_13 : f32
          linalg.yield %10 : f32
        }
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

