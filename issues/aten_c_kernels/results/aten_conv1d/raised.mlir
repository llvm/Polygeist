#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4 + d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_conv1d(%arg0: memref<?x3x16xf32>, %arg1: memref<?x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x4x14xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c14 = arith.constant 14 : index
    %c3 = arith.constant 3 : index
    %subview = memref.subview %arg2[0] [%c4] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_0 = memref.subview %arg3[0, 0, 0] [%c2, %c4, %c14] [1, 1, 1] : memref<?x4x14xf32> to memref<?x?x?xf32, strided<[56, 14, 1]>>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<?xf32, strided<[1]>>) outs(%subview_0 : memref<?x?x?xf32, strided<[56, 14, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %0 = polygeist.submap(%arg0, %c2, %c4, %c14, %c3, %c3) {map = #map2} : (memref<?x3x16xf32>, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
    %subview_1 = memref.subview %arg1[0, 0, 0] [%c4, %c3, %c3] [1, 1, 1] : memref<?x3x3xf32> to memref<?x?x?xf32, strided<[9, 3, 1]>>
    %subview_2 = memref.subview %arg3[0, 0, 0] [%c2, %c4, %c14] [1, 1, 1] : memref<?x4x14xf32> to memref<?x?x?xf32, strided<[56, 14, 1]>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %subview_1 : memref<?x?x?x?x?xf32>, memref<?x?x?xf32, strided<[9, 3, 1]>>) outs(%subview_2 : memref<?x?x?xf32, strided<[56, 14, 1]>>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %1 = arith.mulf %in, %in_3 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}

