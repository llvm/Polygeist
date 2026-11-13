#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_gradient_float_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c126 = arith.constant 126 : index
    %0 = affine.load %arg0[1] : memref<?xf32>
    %1 = affine.load %arg0[0] : memref<?xf32>
    %2 = arith.subf %0, %1 : f32
    %3 = affine.load %arg1[1] : memref<?xf32>
    %4 = affine.load %arg1[0] : memref<?xf32>
    %5 = arith.subf %3, %4 : f32
    %6 = arith.divf %2, %5 : f32
    affine.store %6, %arg2[0] : memref<?xf32>
    %subview = memref.subview %arg0[2] [%c126] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 2>>
    %subview_0 = memref.subview %arg0[0] [%c126] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_1 = memref.subview %arg1[2] [%c126] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 2>>
    %subview_2 = memref.subview %arg1[0] [%c126] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_3 = memref.subview %arg2[1] [%c126] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 1>>
    linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%subview, %subview_0, %subview_1, %subview_2 : memref<?xf32, strided<[1], offset: 2>>, memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1], offset: 2>>, memref<?xf32, strided<[1]>>) outs(%subview_3 : memref<?xf32, strided<[1], offset: 1>>) {
    ^bb0(%in: f32, %in_4: f32, %in_5: f32, %in_6: f32, %out: f32):
      %14 = arith.subf %in, %in_4 : f32
      %15 = arith.subf %in_5, %in_6 : f32
      %16 = arith.divf %14, %15 : f32
      linalg.yield %16 : f32
    }
    %7 = affine.load %arg0[127] : memref<?xf32>
    %8 = affine.load %arg0[126] : memref<?xf32>
    %9 = arith.subf %7, %8 : f32
    %10 = affine.load %arg1[127] : memref<?xf32>
    %11 = affine.load %arg1[126] : memref<?xf32>
    %12 = arith.subf %10, %11 : f32
    %13 = arith.divf %9, %12 : f32
    affine.store %13, %arg2[127] : memref<?xf32>
    return
  }
}

