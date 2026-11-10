module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @neg_hist_saturate(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i8 = arith.constant 1 : i8
    %c255_i32 = arith.constant 255 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to %0 {
      %1 = affine.load %arg1[%arg3] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = memref.load %arg2[%2] : memref<?xi8>
      %4 = arith.extui %3 : i8 to i32
      %5 = arith.cmpi slt, %4, %c255_i32 : i32
      scf.if %5 {
        %6 = affine.load %arg1[%arg3] : memref<?xi32>
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg2[%7] : memref<?xi8>
        %9 = arith.addi %8, %c1_i8 : i8
        memref.store %9, %arg2[%7] : memref<?xi8>
      }
    }
    return
  }
}
