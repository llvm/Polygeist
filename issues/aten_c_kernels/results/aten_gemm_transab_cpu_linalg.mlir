#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_gemm_transab_cpu(%arg0: memref<?x40xf32>, %arg1: memref<?x48xf32>, %arg2: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c48 = arith.constant 48 : index
    %c40 = arith.constant 40 : index
    %subview = memref.subview %arg0[0, 0] [%c40, %c32] [1, 1] : memref<?x40xf32> to memref<?x?xf32, strided<[40, 1]>>
    %subview_0 = memref.subview %arg1[0, 0] [%c48, %c40] [1, 1] : memref<?x48xf32> to memref<?x?xf32, strided<[48, 1]>>
    %subview_1 = memref.subview %arg2[0, 0] [%c32, %c48] [1, 1] : memref<?x48xf32> to memref<?x?xf32, strided<[48, 1]>>
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_0 : memref<?x?xf32, strided<[40, 1]>>, memref<?x?xf32, strided<[48, 1]>>) outs(%subview_1 : memref<?x?xf32, strided<[48, 1]>>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %0 = arith.mulf %in, %in_2 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
}

