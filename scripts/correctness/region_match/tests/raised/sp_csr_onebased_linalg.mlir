module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @sp_csr_onebased(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %0 {
      %1 = affine.load %arg1[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %c-1_i32 : i32
      %3 = affine.load %arg1[%arg6 + 1] : memref<?xi32>
      %4 = arith.addi %3, %c-1_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %2 : i32 to index
      %7 = scf.for %arg7 = %6 to %5 step %c1 iter_args(%arg8 = %cst) -> (f64) {
        %8 = memref.load %arg3[%arg7] : memref<?xf64>
        %9 = memref.load %arg2[%arg7] : memref<?xi32>
        %10 = arith.addi %9, %c-1_i32 : i32
        %11 = arith.index_cast %10 : i32 to index
        %12 = memref.load %arg4[%11] : memref<?xf64>
        %13 = arith.mulf %8, %12 : f64
        %14 = arith.addf %arg8, %13 : f64
        scf.yield %14 : f64
      }
      affine.store %7, %arg5[%arg6] : memref<?xf64>
    }
    return
  }
}

