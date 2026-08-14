#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 - 9)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad2d(%arg0: memref<?x8x8xf32>, %arg1: memref<?x10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c-1_i32 = arith.constant -1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %subview = memref.subview %arg1[0, 0, 0] [%c3, %c10, %c10] [1, 1, 1] : memref<?x10x10xf32> to memref<?x?x?xf32, strided<[100, 10, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf32, strided<[100, 10, 1]>>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.cmpi eq, %1, %c0 : index
      %4 = affine.apply #map1(%1)
      %5 = arith.cmpi eq, %4, %c0 : index
      %6 = arith.addi %2, %c-1_i32 : i32
      %7 = arith.select %5, %c7_i32, %6 : i32
      %8 = arith.select %3, %c0_i32, %7 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = linalg.index 2 : index
      %11 = arith.index_cast %10 : index to i32
      %12 = arith.cmpi eq, %10, %c0 : index
      %13 = affine.apply #map1(%10)
      %14 = arith.cmpi eq, %13, %c0 : index
      %15 = arith.addi %11, %c-1_i32 : i32
      %16 = arith.select %14, %c7_i32, %15 : i32
      %17 = arith.select %12, %c0_i32, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg0[%0, %9, %18] : memref<?x8x8xf32>
      linalg.yield %19 : f32
    }
    return
  }
}

