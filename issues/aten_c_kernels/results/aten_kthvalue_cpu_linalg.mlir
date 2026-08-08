#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_kthvalue_cpu(%arg0: memref<?x63xf32>, %arg1: i32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %0 = arith.index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to 16 {
      %1 = affine.apply #map()[%0]
      %alloca = memref.alloca(%1) : memref<?xi32>
      affine.for %arg4 = 0 to #map()[%0] {
        %2 = arith.index_cast %arg4 : index to i32
        affine.store %2, %alloca[%arg4] : memref<?xi32>
        affine.for %arg5 = #map1(%arg4) to 63 {
          %7 = affine.load %alloca[%arg4] : memref<?xi32>
          %8 = arith.index_cast %arg5 : index to i32
          %9 = affine.load %arg0[%arg3, %arg5] : memref<?x63xf32>
          %10 = arith.index_cast %7 : i32 to index
          %11 = memref.load %arg0[%arg3, %10] : memref<?x63xf32>
          %12 = arith.cmpf olt, %9, %11 : f32
          %13 = arith.select %12, %8, %7 : i32
          affine.store %13, %alloca[%arg4] : memref<?xi32>
        }
        %3 = affine.load %alloca[%arg4] : memref<?xi32>
        %4 = affine.load %arg0[%arg3, %arg4] : memref<?x63xf32>
        %5 = arith.index_cast %3 : i32 to index
        %6 = memref.load %arg0[%arg3, %5] : memref<?x63xf32>
        affine.store %6, %arg0[%arg3, %arg4] : memref<?x63xf32>
        memref.store %4, %arg0[%arg3, %5] : memref<?x63xf32>
      }
    }
    %subview = memref.subview %arg0[0, %0] [%c16, 1] [1, 1] : memref<?x63xf32> to memref<?xf32, strided<[63], offset: ?>>
    %subview_0 = memref.subview %arg2[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[63], offset: ?>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

