#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_median_indices_cpu(%arg0: memref<?x63xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c31_i32 = arith.constant 31 : i32
    affine.for %arg3 = 0 to 16 {
      %alloca = memref.alloca(%c32) : memref<?xi32>
      affine.for %arg4 = 0 to 32 {
        %0 = arith.index_cast %arg4 : index to i32
        affine.store %0, %alloca[%arg4] : memref<?xi32>
        affine.for %arg5 = #map(%arg4) to 63 {
          %5 = affine.load %alloca[%arg4] : memref<?xi32>
          %6 = arith.index_cast %arg5 : index to i32
          %7 = affine.load %arg0[%arg3, %arg5] : memref<?x63xf32>
          %8 = arith.index_cast %5 : i32 to index
          %9 = memref.load %arg0[%arg3, %8] : memref<?x63xf32>
          %10 = arith.cmpf olt, %7, %9 : f32
          %11 = arith.select %10, %6, %5 : i32
          affine.store %11, %alloca[%arg4] : memref<?xi32>
        }
        %1 = affine.load %alloca[%arg4] : memref<?xi32>
        %2 = affine.load %arg0[%arg3, %arg4] : memref<?x63xf32>
        %3 = arith.index_cast %1 : i32 to index
        %4 = memref.load %arg0[%arg3, %3] : memref<?x63xf32>
        affine.store %4, %arg0[%arg3, %arg4] : memref<?x63xf32>
        memref.store %2, %arg0[%arg3, %3] : memref<?x63xf32>
      }
    }
    %subview = memref.subview %arg0[0, 31] [%c16, 1] [1, 1] : memref<?x63xf32> to memref<?xf32, strided<[63], offset: 31>>
    %subview_0 = memref.subview %arg1[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[63], offset: 31>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%arg2 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c31_i32 : i32
    }
    return
  }
}

