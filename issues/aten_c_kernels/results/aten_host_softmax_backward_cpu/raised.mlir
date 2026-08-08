#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_host_softmax_backward_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 32 {
      %alloca = memref.alloca() : memref<f32>
      affine.store %cst, %alloca[] : memref<f32>
      %subview = memref.subview %arg0[%arg3, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg3, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %alloca[] [] [] : memref<f32> to memref<f32, strided<[]>>
      linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"]} ins(%subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_1 : memref<f32, strided<[]>>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %1 = arith.mulf %in, %in_5 : f32
        %2 = arith.addf %out, %1 : f32
        linalg.yield %2 : f32
      }
      %0 = affine.load %alloca[] : memref<f32>
      %subview_2 = memref.subview %arg1[%arg3, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_3 = memref.subview %arg0[%arg3, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %arg2[%arg3, 0] [1, %c64] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%subview_2, %subview_3 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_4 : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %1 = arith.subf %in_5, %0 : f32
        %2 = arith.mulf %in, %1 : f32
        linalg.yield %2 : f32
      }
    } {polygeist.was_parallel}
    return
  }
}

