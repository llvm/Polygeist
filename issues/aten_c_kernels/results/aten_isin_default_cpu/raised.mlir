#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_isin_default_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4096 = arith.constant 4096 : index
    %c257 = arith.constant 257 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %alloca_0 = memref.alloca(%c4096) : memref<?xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_0 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %subview = memref.subview %arg1[0] [%c257] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %1 = polygeist.submap(%arg0, %c4096) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %subview_1 = memref.subview %1[0] [%c4096] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_2 = memref.subview %alloca_0[0] [%c4096] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "reduction"]} ins(%subview_1, %subview : memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>) outs(%subview_2 : memref<?xi32, strided<[1]>>) {
    ^bb0(%in: f32, %in_3: f32, %out: i32):
      %2 = arith.cmpf oeq, %in, %in_3 : f32
      %3 = arith.extui %2 : i1 to i32
      %4 = arith.ori %out, %3 : i32
      linalg.yield %4 : i32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca_0 : memref<?xi32>) outs(%arg2 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

