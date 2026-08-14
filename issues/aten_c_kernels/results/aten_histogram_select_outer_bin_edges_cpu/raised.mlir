#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_histogram_select_outer_bin_edges_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4095 = arith.constant 4095 : index
    %0 = affine.load %arg0[0] : memref<?xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_0[] : memref<f32>
    %subview = memref.subview %arg0[1] [%c4095] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 1>>
    %subview_1 = memref.subview %alloca[] [] [] : memref<f32> to memref<f32, strided<[]>>
    %subview_2 = memref.subview %alloca_0[] [] [] : memref<f32> to memref<f32, strided<[]>>
    linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xf32, strided<[1], offset: 1>>) outs(%subview_1, %subview_2 : memref<f32, strided<[]>>, memref<f32, strided<[]>>) {
    ^bb0(%in: f32, %out: f32, %out_3: f32):
      %3 = arith.cmpf olt, %in, %out_3 : f32
      %4 = arith.select %3, %in, %out_3 : f32
      %5 = arith.cmpf ogt, %in, %out : f32
      %6 = arith.select %5, %in, %out : f32
      linalg.yield %6, %4 : f32, f32
    }
    %1 = affine.load %alloca[] : memref<f32>
    %2 = affine.load %alloca_0[] : memref<f32>
    affine.store %2, %arg1[0] : memref<?xf32>
    affine.store %1, %arg2[0] : memref<?xf32>
    return
  }
}

