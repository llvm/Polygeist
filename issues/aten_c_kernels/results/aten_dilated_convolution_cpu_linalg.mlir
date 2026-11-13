#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4 * 2 + d1, d5 * 2 + d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dilated_convolution_cpu(%arg0: memref<?x16x16xf32>, %arg1: memref<?x2x3x3xf32>, %arg2: memref<?x12x12xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg2[0, 0, 0] [%c3, %c12, %c12] [1, 1, 1] : memref<?x12x12xf32> to memref<?x?x?xf32, strided<[144, 12, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf32, strided<[144, 12, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %0 = polygeist.submap(%arg0, %c3, %c12, %c12, %c2, %c3, %c3) {map = #map1} : (memref<?x16x16xf32>, index, index, index, index, index, index) -> memref<?x?x?x?x?x?xf32>
    %subview_0 = memref.subview %arg1[0, 0, 0, 0] [%c3, %c2, %c3, %c3] [1, 1, 1, 1] : memref<?x2x3x3xf32> to memref<?x?x?x?xf32, strided<[18, 9, 3, 1]>>
    %subview_1 = memref.subview %arg2[0, 0, 0] [%c3, %c12, %c12] [1, 1, 1] : memref<?x12x12xf32> to memref<?x?x?xf32, strided<[144, 12, 1]>>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %subview_0 : memref<?x?x?x?x?x?xf32>, memref<?x?x?x?xf32, strided<[18, 9, 3, 1]>>) outs(%subview_1 : memref<?x?x?xf32, strided<[144, 12, 1]>>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}

