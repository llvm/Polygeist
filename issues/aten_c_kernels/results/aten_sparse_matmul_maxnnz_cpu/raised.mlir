module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_matmul_maxnnz_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg5 = 0 to 64 {
      %0 = affine.load %arg0[%arg5] : memref<?xi32>
      %1 = affine.load %arg0[%arg5 + 1] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %0 : i32 to index
      %4 = scf.for %arg6 = %3 to %2 step %c1 iter_args(%arg7 = %c0_i32) -> (i32) {
        %5 = memref.load %arg1[%arg6] : memref<?xi32>
        %6 = arith.addi %5, %c1_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg2[%7] : memref<?xi32>
        %9 = arith.index_cast %5 : i32 to index
        %10 = memref.load %arg2[%9] : memref<?xi32>
        %11 = arith.subi %8, %10 : i32
        %12 = arith.addi %arg7, %11 : i32
        scf.yield %12 : i32
      }
      affine.store %4, %arg4[%arg5] : memref<?xi32>
    }
    return
  }
}

