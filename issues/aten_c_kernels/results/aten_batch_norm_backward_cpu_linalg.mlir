#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_backward_cpu(%arg0: memref<?x8x32xf32>, %arg1: memref<?x8x32xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x8x32xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 1.280000e+02 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca(%c8) : memref<?xf32>
    %alloca_1 = memref.alloca(%c8) : memref<?xf32>
    affine.for %arg8 = 0 to 8 {
      %0 = affine.load %arg2[%arg8] : memref<?xf32>
      affine.store %cst_0, %alloca[%arg8] : memref<?xf32>
      affine.store %cst_0, %alloca_1[%arg8] : memref<?xf32>
      %alloca_2 = memref.alloca(%c4) : memref<?xf32>
      %alloca_3 = memref.alloca(%c4) : memref<?xf32>
      affine.for %arg9 = 0 to 4 {
        %5 = affine.load %alloca[%arg8] : memref<?xf32>
        %6 = affine.load %alloca_1[%arg8] : memref<?xf32>
        affine.store %5, %alloca_2[%arg9] : memref<?xf32>
        affine.store %6, %alloca_3[%arg9] : memref<?xf32>
        %subview_9 = memref.subview %arg0[%arg9, %arg8, 0] [1, 1, %c32] [1, 1, 1] : memref<?x8x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_10 = memref.subview %arg1[%arg9, %arg8, 0] [1, 1, %c32] [1, 1, 1] : memref<?x8x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_11 = memref.subview %alloca_2[%arg9] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
        %subview_12 = memref.subview %alloca_3[%arg9] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["reduction"]} ins(%subview_9, %subview_10 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_11, %subview_12 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %in_13: f32, %out: f32, %out_14: f32):
          %9 = arith.addf %out_14, %in : f32
          %10 = arith.subf %in_13, %0 : f32
          %11 = arith.mulf %in, %10 : f32
          %12 = arith.addf %out, %11 : f32
          linalg.yield %12, %9 : f32, f32
        }
        %7 = affine.load %alloca_2[%arg9] : memref<?xf32>
        %8 = affine.load %alloca_3[%arg9] : memref<?xf32>
        affine.store %7, %alloca[%arg8] : memref<?xf32>
        affine.store %8, %alloca_1[%arg8] : memref<?xf32>
      }
      %1 = affine.load %alloca[%arg8] : memref<?xf32>
      %2 = affine.load %alloca_1[%arg8] : memref<?xf32>
      affine.store %2, %arg7[%arg8] : memref<?xf32>
      %3 = affine.load %arg3[%arg8] : memref<?xf32>
      %4 = arith.mulf %1, %3 : f32
      affine.store %4, %arg6[%arg8] : memref<?xf32>
      %subview = memref.subview %arg4[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_4 = memref.subview %arg3[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_5 = memref.subview %arg0[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : memref<?x8x32xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %subview_6 = memref.subview %arg1[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : memref<?x8x32xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %subview_7 = memref.subview %arg2[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_8 = memref.subview %arg5[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : memref<?x8x32xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map2, #map2, #map3, #map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_4, %subview_5, %subview_6, %subview_7 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>, memref<?x?xf32, strided<[256, 1], offset: ?>>, memref<?x?xf32, strided<[256, 1], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_8 : memref<?x?xf32, strided<[256, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %in_12: f32, %out: f32):
        %5 = arith.mulf %in, %in_9 : f32
        %6 = arith.divf %5, %cst : f32
        %7 = arith.mulf %in_10, %cst : f32
        %8 = arith.subf %7, %2 : f32
        %9 = arith.subf %in_11, %in_12 : f32
        %10 = arith.mulf %9, %in_9 : f32
        %11 = arith.mulf %10, %in_9 : f32
        %12 = arith.mulf %11, %1 : f32
        %13 = arith.subf %8, %12 : f32
        %14 = arith.mulf %6, %13 : f32
        linalg.yield %14 : f32
      }
    } {polygeist.was_parallel}
    return
  }
}

