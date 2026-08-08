#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_or_reduce_cpu(%arg0: memref<?x64xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %subview = memref.subview %arg0[0, 0] [%c32, %c64] [1, 1] : memref<?x64xi32> to memref<?x?xi32, strided<[64, 1]>>
    %subview_0 = memref.subview %arg1[0] [%c32] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<?x?xi32, strided<[64, 1]>>) outs(%subview_0 : memref<?xi32, strided<[1]>>) {
    ^bb0(%in: i32, %out: i32):
      %0 = arith.cmpi ne, %out, %c0_i32 : i32
      %1 = arith.cmpi ne, %in, %c0_i32 : i32
      %2 = arith.select %0, %true, %1 : i1
      %3 = arith.extsi %2 : i1 to i32
      linalg.yield %3 : i32
    }
    return
  }
}

