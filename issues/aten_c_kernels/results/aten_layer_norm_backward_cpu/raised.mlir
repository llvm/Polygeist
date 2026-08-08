#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_layer_norm_backward_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x64xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg6 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg7 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    affine.for %arg8 = 0 to 16 {
      %alloca = memref.alloca() : memref<f32>
      affine.store %cst_0, %alloca[] : memref<f32>
      %alloca_1 = memref.alloca() : memref<f32>
      affine.store %cst_0, %alloca_1[] : memref<f32>
      %subview = memref.subview %arg0[%arg8, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_2 = memref.subview %arg4[0] [%c64] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %subview_3 = memref.subview %arg1[%arg8, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %arg2[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_5 = memref.subview %arg3[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_6 = memref.subview %arg0[%arg8, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_7 = memref.subview %arg6[0] [%c64] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %subview_8 = memref.subview %arg7[0] [%c64] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %subview_9 = memref.subview %alloca[] [] [] : memref<f32> to memref<f32, strided<[]>>
      %subview_10 = memref.subview %alloca_1[] [] [] : memref<f32> to memref<f32, strided<[]>>
      linalg.generic {indexing_maps = [#map, #map, #map, #map1, #map1, #map, #map, #map, #map1, #map1], iterator_types = ["reduction"]} ins(%subview, %subview_2, %subview_3, %subview_4, %subview_5, %subview_6 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1], offset: ?>>, memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_7, %subview_8, %subview_9, %subview_10 : memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>, memref<f32, strided<[]>>, memref<f32, strided<[]>>) {
      ^bb0(%in: f32, %in_17: f32, %in_18: f32, %in_19: f32, %in_20: f32, %in_21: f32, %out: f32, %out_22: f32, %out_23: f32, %out_24: f32):
        %2 = arith.mulf %in, %in_17 : f32
        %3 = arith.addf %out_24, %2 : f32
        %4 = arith.subf %in_18, %in_19 : f32
        %5 = arith.mulf %2, %4 : f32
        %6 = arith.addf %out_23, %5 : f32
        %7 = arith.mulf %in, %4 : f32
        %8 = arith.mulf %7, %in_20 : f32
        %9 = arith.addf %out, %8 : f32
        %10 = arith.addf %out_22, %in_21 : f32
        linalg.yield %9, %10, %6, %3 : f32, f32, f32, f32
      }
      %0 = affine.load %alloca[] : memref<f32>
      %1 = affine.load %alloca_1[] : memref<f32>
      %subview_11 = memref.subview %arg0[%arg8, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %arg4[0] [%c64] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %subview_13 = memref.subview %arg3[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_14 = memref.subview %arg1[%arg8, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_15 = memref.subview %arg2[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_16 = memref.subview %arg5[%arg8, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map1, #map, #map1, #map], iterator_types = ["parallel"]} ins(%subview_11, %subview_12, %subview_13, %subview_14, %subview_15 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1]>>, memref<f32, strided<[], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_16 : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_17: f32, %in_18: f32, %in_19: f32, %in_20: f32, %out: f32):
        %2 = arith.mulf %in, %in_17 : f32
        %3 = arith.divf %in_18, %cst : f32
        %4 = arith.mulf %2, %cst : f32
        %5 = arith.subf %4, %1 : f32
        %6 = arith.subf %in_19, %in_20 : f32
        %7 = arith.mulf %6, %in_18 : f32
        %8 = arith.mulf %7, %in_18 : f32
        %9 = arith.mulf %8, %0 : f32
        %10 = arith.subf %5, %9 : f32
        %11 = arith.mulf %3, %10 : f32
        linalg.yield %11 : f32
      }
    }
    return
  }
}

