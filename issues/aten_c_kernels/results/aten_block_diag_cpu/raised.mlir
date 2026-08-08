#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1 + d0 * 16, d2 + d0 * 16)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_block_diag_cpu(%arg0: memref<?x16x16xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg1[0, 0] [%c64, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[64, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %subview_0 = memref.subview %arg0[0, 0, 0] [%c4, %c16, %c16] [1, 1, 1] : memref<?x16x16xf32> to memref<?x?x?xf32, strided<[256, 16, 1]>>
    %0 = polygeist.submap(%arg1, %c4, %c16, %c16) {map = #map1} : (memref<?x64xf32>, index, index, index) -> memref<?x?x?xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_0 : memref<?x?x?xf32, strided<[256, 16, 1]>>) outs(%0 : memref<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

