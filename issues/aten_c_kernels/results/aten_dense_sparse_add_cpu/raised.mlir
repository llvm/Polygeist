#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dense_sparse_add_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %subview = memref.subview %arg0[0, 0] [%c64, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_0 = memref.subview %arg4[0, 0] [%c64, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<?x?xf32, strided<[64, 1]>>) outs(%subview_0 : memref<?x?xf32, strided<[64, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    affine.for %arg5 = 0 to 512 {
      %0 = affine.load %arg1[%arg5] : memref<?xi32>
      %1 = arith.index_cast %0 : i32 to index
      %2 = affine.load %arg2[%arg5] : memref<?xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = affine.load %arg3[%arg5] : memref<?xf32>
      %5 = memref.load %arg4[%1, %3] : memref<?x64xf32>
      %6 = arith.addf %5, %4 : f32
      memref.store %6, %arg4[%1, %3] : memref<?x64xf32>
    }
    return
  }
}

