#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5 + d2, d6 + d3, d7 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_conv_transpose3d_grad_weight_cpu(%arg0: memref<?x6x7x8xf32>, %arg1: memref<?x8x9x10xf32>, %arg2: memref<?x3x3x3x3xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "polygeist.memref2pointer"(%arg2) : (memref<?x3x3x3x3xf32>) -> !llvm.ptr
    affine.for %arg3 = 0 to 162 {
      %2 = arith.index_cast %arg3 : index to i32
      %3 = llvm.getelementptr %0[%2] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %3 : f32, !llvm.ptr
    }
    %subview = memref.subview %arg0[0, 0, 0, 0] [%c2, %c6, %c7, %c8] [1, 1, 1, 1] : memref<?x6x7x8xf32> to memref<?x?x?x?xf32, strided<[336, 56, 8, 1]>>
    %1 = polygeist.submap(%arg1, %c2, %c3, %c3, %c3, %c3, %c6, %c7, %c8) {map = #map} : (memref<?x8x9x10xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?x?xf32>
    %subview_0 = memref.subview %arg2[0, 0, 0, 0, 0] [%c2, %c3, %c3, %c3, %c3] [1, 1, 1, 1, 1] : memref<?x3x3x3x3xf32> to memref<?x?x?x?x?xf32, strided<[81, 27, 9, 3, 1]>>
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview, %1 : memref<?x?x?x?xf32, strided<[336, 56, 8, 1]>>, memref<?x?x?x?x?x?x?x?xf32>) outs(%subview_0 : memref<?x?x?x?x?xf32, strided<[81, 27, 9, 3, 1]>>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in, %in_1 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    return
  }
}

