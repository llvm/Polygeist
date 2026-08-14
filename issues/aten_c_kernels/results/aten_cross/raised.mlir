#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cross(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %subview = memref.subview %arg0[0, 1] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 1>>
    %subview_0 = memref.subview %arg1[0, 2] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 2>>
    %subview_1 = memref.subview %arg0[0, 2] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 2>>
    %subview_2 = memref.subview %arg1[0, 1] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 1>>
    %subview_3 = memref.subview %arg2[0, 0] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3]>>
    linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%subview, %subview_0, %subview_1, %subview_2 : memref<?xf32, strided<[3], offset: 1>>, memref<?xf32, strided<[3], offset: 2>>, memref<?xf32, strided<[3], offset: 2>>, memref<?xf32, strided<[3], offset: 1>>) outs(%subview_3 : memref<?xf32, strided<[3]>>) {
    ^bb0(%in: f32, %in_14: f32, %in_15: f32, %in_16: f32, %out: f32):
      %0 = arith.mulf %in, %in_14 : f32
      %1 = arith.mulf %in_15, %in_16 : f32
      %2 = arith.subf %0, %1 : f32
      linalg.yield %2 : f32
    }
    %subview_4 = memref.subview %arg0[0, 2] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 2>>
    %subview_5 = memref.subview %arg1[0, 0] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3]>>
    %subview_6 = memref.subview %arg0[0, 0] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3]>>
    %subview_7 = memref.subview %arg1[0, 2] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 2>>
    %subview_8 = memref.subview %arg2[0, 1] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 1>>
    linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%subview_4, %subview_5, %subview_6, %subview_7 : memref<?xf32, strided<[3], offset: 2>>, memref<?xf32, strided<[3]>>, memref<?xf32, strided<[3]>>, memref<?xf32, strided<[3], offset: 2>>) outs(%subview_8 : memref<?xf32, strided<[3], offset: 1>>) {
    ^bb0(%in: f32, %in_14: f32, %in_15: f32, %in_16: f32, %out: f32):
      %0 = arith.mulf %in, %in_14 : f32
      %1 = arith.mulf %in_15, %in_16 : f32
      %2 = arith.subf %0, %1 : f32
      linalg.yield %2 : f32
    }
    %subview_9 = memref.subview %arg0[0, 0] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3]>>
    %subview_10 = memref.subview %arg1[0, 1] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 1>>
    %subview_11 = memref.subview %arg0[0, 1] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 1>>
    %subview_12 = memref.subview %arg1[0, 0] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3]>>
    %subview_13 = memref.subview %arg2[0, 2] [%c64, 1] [1, 1] : memref<?x3xf32> to memref<?xf32, strided<[3], offset: 2>>
    linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%subview_9, %subview_10, %subview_11, %subview_12 : memref<?xf32, strided<[3]>>, memref<?xf32, strided<[3], offset: 1>>, memref<?xf32, strided<[3], offset: 1>>, memref<?xf32, strided<[3]>>) outs(%subview_13 : memref<?xf32, strided<[3], offset: 2>>) {
    ^bb0(%in: f32, %in_14: f32, %in_15: f32, %in_16: f32, %out: f32):
      %0 = arith.mulf %in, %in_14 : f32
      %1 = arith.mulf %in_15, %in_16 : f32
      %2 = arith.subf %0, %1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}

