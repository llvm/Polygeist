#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cdist_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?x12xf32>, %arg3: memref<?x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg3[0, 0] [%c16, %c32] [1, 1] : memref<?x32xf32> to memref<?x?xf32, strided<[32, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[32, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 12 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst, %alloca[] : memref<f32>
        %subview_0 = memref.subview %arg0[%arg4, 0] [1, %c32] [1, 1] : memref<?x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg5, 0] [1, %c32] [1, 1] : memref<?x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_2 = memref.subview %alloca[] [] [] : memref<f32> to memref<f32, strided<[]>>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_0, %subview_1 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_2 : memref<f32, strided<[]>>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %6 = arith.subf %in, %in_6 : f32
          %7 = arith.mulf %6, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        }
        %0 = affine.load %alloca[] : memref<f32>
        %1 = math.sqrt %0 : f32
        %2 = arith.cmpf oeq, %1, %cst : f32
        %3 = affine.load %arg2[%arg4, %arg5] : memref<?x12xf32>
        %4 = arith.divf %3, %1 : f32
        %5 = arith.select %2, %cst, %4 : f32
        %subview_3 = memref.subview %arg0[%arg4, 0] [1, %c32] [1, 1] : memref<?x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_4 = memref.subview %arg1[%arg5, 0] [1, %c32] [1, 1] : memref<?x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %subview_5 = memref.subview %arg3[%arg4, 0] [1, %c32] [1, 1] : memref<?x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_5 : memref<?xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %6 = arith.subf %in, %in_6 : f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        }
      }
    }
    return
  }
}

