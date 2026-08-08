#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_stack_serial_cpu(%arg0: memref<?x16x32xf32>, %arg1: memref<?x4x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %subview = memref.subview %arg0[0, 0, 0] [%c4, %c16, %c32] [1, 1, 1] : memref<?x16x32xf32> to memref<?x?x?xf32, strided<[512, 32, 1]>>
    %subview_0 = memref.subview %arg1[0, 0, 0] [%c16, %c4, %c32] [1, 1, 1] : memref<?x4x32xf32> to memref<?x?x?xf32, strided<[128, 32, 1]>>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<?x?x?xf32, strided<[512, 32, 1]>>) outs(%subview_0 : memref<?x?x?xf32, strided<[128, 32, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

