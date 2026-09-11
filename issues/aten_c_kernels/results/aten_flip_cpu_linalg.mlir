#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flip_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg2, %c0_i32 : i32
    %1 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %subview = memref.subview %arg1[0, 0] [%c32, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[64, 1]>>) {
    ^bb0(%out: f32):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.subi %c31_i32, %3 : i32
      %5 = arith.select %0, %4, %3 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %7 : index to i32
      %9 = arith.subi %c63_i32, %8 : i32
      %10 = arith.select %1, %9, %8 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = memref.load %arg0[%6, %11] : memref<?x64xf32>
      linalg.yield %12 : f32
    }
    return
  }
}

