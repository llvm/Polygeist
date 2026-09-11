#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_transform_bias_rescale_qkv_cpu(%arg0: memref<?x16x3x4x8xf32>, %arg1: memref<?x4x8xf32>, %arg2: f32, %arg3: memref<?x4x16x8xf32>, %arg4: memref<?x4x16x8xf32>, %arg5: memref<?x4x16x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %subview = memref.subview %arg0[0, 0, 0, 0, 0] [%c2, %c16, 1, %c4, %c8] [1, 1, 1, 1, 1] : memref<?x16x3x4x8xf32> to memref<?x?x?x?xf32, strided<[1536, 96, 8, 1]>>
    %subview_0 = memref.subview %arg1[0, 0, 0] [1, %c4, %c8] [1, 1, 1] : memref<?x4x8xf32> to memref<?x?xf32, strided<[8, 1]>>
    %subview_1 = memref.subview %arg3[0, 0, 0, 0] [%c2, %c4, %c16, %c8] [1, 1, 1, 1] : memref<?x4x16x8xf32> to memref<?x?x?x?xf32, strided<[512, 128, 8, 1]>>
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview, %subview_0 : memref<?x?x?x?xf32, strided<[1536, 96, 8, 1]>>, memref<?x?xf32, strided<[8, 1]>>) outs(%subview_1 : memref<?x?x?x?xf32, strided<[512, 128, 8, 1]>>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %0 = arith.addf %in, %in_8 : f32
      %1 = arith.mulf %0, %arg2 : f32
      linalg.yield %1 : f32
    }
    %subview_2 = memref.subview %arg0[0, 0, 1, 0, 0] [%c2, %c16, 1, %c4, %c8] [1, 1, 1, 1, 1] : memref<?x16x3x4x8xf32> to memref<?x?x?x?xf32, strided<[1536, 96, 8, 1], offset: 32>>
    %subview_3 = memref.subview %arg1[1, 0, 0] [1, %c4, %c8] [1, 1, 1] : memref<?x4x8xf32> to memref<?x?xf32, strided<[8, 1], offset: 32>>
    %subview_4 = memref.subview %arg4[0, 0, 0, 0] [%c2, %c4, %c16, %c8] [1, 1, 1, 1] : memref<?x4x16x8xf32> to memref<?x?x?x?xf32, strided<[512, 128, 8, 1]>>
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_2, %subview_3 : memref<?x?x?x?xf32, strided<[1536, 96, 8, 1], offset: 32>>, memref<?x?xf32, strided<[8, 1], offset: 32>>) outs(%subview_4 : memref<?x?x?x?xf32, strided<[512, 128, 8, 1]>>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %0 = arith.addf %in, %in_8 : f32
      linalg.yield %0 : f32
    }
    %subview_5 = memref.subview %arg0[0, 0, 2, 0, 0] [%c2, %c16, 1, %c4, %c8] [1, 1, 1, 1, 1] : memref<?x16x3x4x8xf32> to memref<?x?x?x?xf32, strided<[1536, 96, 8, 1], offset: 64>>
    %subview_6 = memref.subview %arg1[2, 0, 0] [1, %c4, %c8] [1, 1, 1] : memref<?x4x8xf32> to memref<?x?xf32, strided<[8, 1], offset: 64>>
    %subview_7 = memref.subview %arg5[0, 0, 0, 0] [%c2, %c4, %c16, %c8] [1, 1, 1, 1] : memref<?x4x16x8xf32> to memref<?x?x?x?xf32, strided<[512, 128, 8, 1]>>
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_5, %subview_6 : memref<?x?x?x?xf32, strided<[1536, 96, 8, 1], offset: 64>>, memref<?x?xf32, strided<[8, 1], offset: 64>>) outs(%subview_7 : memref<?x?x?x?xf32, strided<[512, 128, 8, 1]>>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %0 = arith.addf %in, %in_8 : f32
      linalg.yield %0 : f32
    }
    return
  }
}

