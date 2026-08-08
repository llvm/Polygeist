#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_rowwise_prune_cpu(%arg0: memref<?x32xf32>, %arg1: f32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca(%c64) : memref<?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %subview = memref.subview %arg0[0, 0] [%c64, %c32] [1, 1] : memref<?x32xf32> to memref<?x?xf32, strided<[32, 1]>>
    %subview_0 = memref.subview %alloca[0] [%c64] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<?x?xf32, strided<[32, 1]>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf olt, %in, %cst : f32
      %1 = arith.negf %in : f32
      %2 = arith.select %0, %1, %in : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca : memref<?xf32>) outs(%arg2 : memref<?xi32>) {
    ^bb0(%in: f32, %out: i32):
      %0 = arith.cmpf ogt, %in, %arg1 : f32
      %1 = arith.extui %0 : i1 to i32
      linalg.yield %1 : i32
    }
    return
  }
}

