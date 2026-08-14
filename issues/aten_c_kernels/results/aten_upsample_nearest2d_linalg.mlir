#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest2d(%arg0: memref<?x4x8x8xf32>, %arg1: memref<?x4x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c-1 = arith.constant -1 : index
    %subview = memref.subview %arg1[0, 0, 0, 0] [%c2, %c4, %c16, %c16] [1, 1, 1, 1] : memref<?x4x16x16xf32> to memref<?x?x?x?xf32, strided<[1024, 256, 16, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?x?xf32, strided<[1024, 256, 16, 1]>>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = arith.cmpi slt, %2, %c0 : index
      %4 = arith.subi %c-1, %2 : index
      %5 = arith.select %3, %4, %2 : index
      %6 = arith.divsi %5, %c2 : index
      %7 = arith.subi %c-1, %6 : index
      %8 = arith.select %3, %7, %6 : index
      %9 = linalg.index 3 : index
      %10 = arith.cmpi slt, %9, %c0 : index
      %11 = arith.subi %c-1, %9 : index
      %12 = arith.select %10, %11, %9 : index
      %13 = arith.divsi %12, %c2 : index
      %14 = arith.subi %c-1, %13 : index
      %15 = arith.select %10, %14, %13 : index
      %16 = memref.load %arg0[%0, %1, %8, %15] : memref<?x4x8x8xf32>
      linalg.yield %16 : f32
    }
    return
  }
}

