module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @sp_csc_i64(%arg0: i64, %arg1: memref<?xi64>, %arg2: memref<?xi64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.index_cast %arg0 : i64 to index
    affine.for %arg6 = 0 to %0 {
      %1 = affine.load %arg1[%arg6] : memref<?xi64>
      %2 = scf.while (%arg7 = %1) : (i64) -> i64 {
        %3 = affine.load %arg1[%arg6 + 1] : memref<?xi64>
        %4 = arith.cmpi slt, %arg7, %3 : i64
        scf.condition(%4) %arg7 : i64
      } do {
      ^bb0(%arg7: i64):
        %3 = arith.index_cast %arg7 : i64 to index
        %4 = memref.load %arg2[%3] : memref<?xi64>
        %5 = arith.index_cast %4 : i64 to index
        %6 = memref.load %arg3[%3] : memref<?xf64>
        %7 = affine.load %arg4[%arg6] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = memref.load %arg5[%5] : memref<?xf64>
        %10 = arith.addf %9, %8 : f64
        memref.store %10, %arg5[%5] : memref<?xf64>
        %11 = arith.addi %arg7, %c1_i64 : i64
        scf.yield %11 : i64
      }
    }
    return
  }
}
