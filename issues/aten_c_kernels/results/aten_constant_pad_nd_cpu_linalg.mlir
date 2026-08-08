#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_constant_pad_nd_cpu(%arg0: memref<?xf32>, %arg1: f32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg2 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %arg1 : f32
    }
    %subview = memref.subview %arg0[0] [%c32] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_0 = memref.subview %arg2[3] [%c32] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 3>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[1]>>) outs(%subview_0 : memref<?xf32, strided<[1], offset: 3>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

