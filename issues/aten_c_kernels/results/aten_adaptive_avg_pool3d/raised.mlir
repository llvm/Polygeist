#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5 + d2 * 2, d6 + d3 * 2, d7 + d4 * 2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool3d(%arg0: memref<?x3x8x8x8xf32>, %arg1: memref<?x3x4x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg1[0, 0, 0, 0, 0] [%c2, %c3, %c4, %c4, %c4] [1, 1, 1, 1, 1] : memref<?x3x4x4x4xf32> to memref<?x?x?x?x?xf32, strided<[192, 64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?x?x?xf32, strided<[192, 64, 16, 4, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    %0 = polygeist.submap(%arg0, %c2, %c3, %c4, %c4, %c4, %c2, %c2, %c2) {map = #map1} : (memref<?x3x8x8x8xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?x?xf32>
    %subview_1 = memref.subview %arg1[0, 0, 0, 0, 0] [%c2, %c3, %c4, %c4, %c4] [1, 1, 1, 1, 1] : memref<?x3x4x4x4xf32> to memref<?x?x?x?x?xf32, strided<[192, 64, 16, 4, 1]>>
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0 : memref<?x?x?x?x?x?x?x?xf32>) outs(%subview_1 : memref<?x?x?x?x?xf32, strided<[192, 64, 16, 4, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.divf %in, %cst : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}

