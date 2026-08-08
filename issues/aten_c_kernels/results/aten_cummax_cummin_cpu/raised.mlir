#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cummax_cummin_cpu(%arg0: memref<?x64xf32>, %arg1: i32, %arg2: memref<?x64xf32>, %arg3: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg1, %c0_i32 : i32
    %alloca = memref.alloca(%c16) : memref<?xi32>
    %alloca_0 = memref.alloca(%c16) : memref<?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %subview = memref.subview %arg0[0, 0] [%c16, 1] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[64]>>
    %subview_1 = memref.subview %alloca_0[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[64]>>) outs(%subview_1 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_2 = memref.subview %arg0[0, 0] [%c16, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_3 = memref.subview %arg0[0, 0] [%c16, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_4 = memref.subview %arg0[0, 0] [%c16, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_5 = memref.subview %arg2[0, 0] [%c16, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_6 = memref.subview %arg3[0, 0] [%c16, %c64] [1, 1] : memref<?x64xi32> to memref<?x?xi32, strided<[64, 1]>>
    %subview_7 = memref.subview %alloca[0] [%c16] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
    %subview_8 = memref.subview %alloca_0[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_2, %subview_3, %subview_4 : memref<?x?xf32, strided<[64, 1]>>, memref<?x?xf32, strided<[64, 1]>>, memref<?x?xf32, strided<[64, 1]>>) outs(%subview_5, %subview_6, %subview_7, %subview_8 : memref<?x?xf32, strided<[64, 1]>>, memref<?x?xi32, strided<[64, 1]>>, memref<?xi32, strided<[1]>>, memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32, %out_11: i32, %out_12: i32, %out_13: f32):
      %1 = linalg.index 1 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.cmpf oge, %in, %out_13 : f32
      %4 = arith.select %0, %3, %false : i1
      %5 = arith.cmpf ole, %in_9, %out_13 : f32
      %6 = arith.select %0, %false, %5 : i1
      %7 = arith.select %4, %true, %6 : i1
      %8 = arith.select %7, %2, %out_12 : i32
      %9 = arith.select %7, %in_10, %out_13 : f32
      linalg.yield %9, %8, %8, %9 : f32, i32, i32, f32
    }
    return
  }
}

