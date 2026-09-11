module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unique_consecutive_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg2[0] : memref<?xi32>
    affine.for %arg3 = 0 to 1024 {
      %0 = affine.load %arg2[0] : memref<?xi32>
      %1 = arith.cmpi eq, %arg3, %c0 : index
      %2 = affine.load %arg0[%arg3] : memref<?xi32>
      %3 = affine.load %arg0[%arg3 - 1] : memref<?xi32>
      %4 = arith.cmpi ne, %2, %3 : i32
      %5 = arith.select %1, %true, %4 : i1
      %6 = scf.if %5 -> (i32) {
        %7 = arith.addi %0, %c1_i32 : i32
        %8 = arith.index_cast %0 : i32 to index
        %9 = affine.load %arg0[%arg3] : memref<?xi32>
        memref.store %9, %arg1[%8] : memref<?xi32>
        scf.yield %7 : i32
      } else {
        scf.yield %0 : i32
      }
      affine.store %6, %arg2[0] : memref<?xi32>
    }
    return
  }
}

