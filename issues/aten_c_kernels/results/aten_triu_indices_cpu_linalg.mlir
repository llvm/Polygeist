#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_triu_indices_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 32 {
      %0 = affine.load %alloca[] : memref<i32>
      %1 = arith.index_cast %arg2 : index to i32
      %2 = arith.subi %c32, %arg2 : index
      %3 = arith.index_cast %0 : i32 to index
      %4 = arith.addi %3, %2 : index
      %5 = arith.index_cast %4 : index to i32
      affine.for %arg3 = #map(%arg2) to 32 {
        %6 = arith.subi %arg3, %arg2 : index
        %7 = arith.addi %3, %6 : index
        %8 = arith.index_cast %arg3 : index to i32
        memref.store %1, %arg0[%7] : memref<?xi32>
        memref.store %8, %arg1[%7] : memref<?xi32>
      }
      affine.store %5, %alloca[] : memref<i32>
    }
    return
  }
}

