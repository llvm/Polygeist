#map = affine_map<(d0, d1) -> (d1 + d0 * 256)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cat_sparse_cpu(%arg0: memref<?x256xi32>, %arg1: memref<?x256xf32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %subview = memref.subview %arg0[0, 0] [%c4, %c256] [1, 1] : memref<?x256xi32> to memref<?x?xi32, strided<[256, 1]>>
    %0 = polygeist.submap(%arg2, %c4, %c256) {map = #map} : (memref<?xi32>, index, index) -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<?x?xi32, strided<[256, 1]>>) outs(%0 : memref<?x?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    %subview_0 = memref.subview %arg1[0, 0] [%c4, %c256] [1, 1] : memref<?x256xf32> to memref<?x?xf32, strided<[256, 1]>>
    %1 = polygeist.submap(%arg3, %c4, %c256) {map = #map} : (memref<?xf32>, index, index) -> memref<?x?xf32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_0 : memref<?x?xf32, strided<[256, 1]>>) outs(%1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

