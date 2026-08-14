#map = affine_map<(d0, d1, d2, d3) -> (d2 + d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_conv_tbc_backward_cpu(%arg0: memref<?x8x24xf32>, %arg1: memref<?x16x24xf32>, %arg2: memref<?x8x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c30 = arith.constant 30 : index
    %c8 = arith.constant 8 : index
    %c24 = arith.constant 24 : index
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [4096], strides: [1] : memref<?x8x16xf32> to memref<4096xf32>
    linalg.fill ins(%cst : f32) outs(%reinterpret_cast : memref<4096xf32>)
    %subview = memref.subview %arg0[0, 0, 0] [%c30, %c8, %c24] [1, 1, 1] : memref<?x8x24xf32> to memref<?x?x?xf32, strided<[192, 24, 1]>>
    %subview_0 = memref.subview %arg1[0, 0, 0] [%c3, %c16, %c24] [1, 1, 1] : memref<?x16x24xf32> to memref<?x?x?xf32, strided<[384, 24, 1]>>
    %0 = polygeist.submap(%arg2, %c30, %c8, %c3, %c16) {map = #map} : (memref<?x8x16xf32>, index, index, index, index) -> memref<30x8x3x16xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} ins(%subview, %subview_0 : memref<?x?x?xf32, strided<[192, 24, 1]>>, memref<?x?x?xf32, strided<[384, 24, 1]>>) outs(%0 : memref<30x8x3x16xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %1 = arith.mulf %in, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}

