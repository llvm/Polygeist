module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @scipy_csr_matvec_beta0(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %0 {
      %1 = affine.load %arg2[%arg7] : memref<?xi32>
      %2 = affine.load %arg2[%arg7 + 1] : memref<?xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %1 : i32 to index
      %5 = scf.for %arg8 = %4 to %3 step %c1 iter_args(%arg9 = %cst) -> (f64) {
        %6 = memref.load %arg4[%arg8] : memref<?xf64>
        %7 = memref.load %arg3[%arg8] : memref<?xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %arg5[%8] : memref<?xf64>
        %10 = arith.mulf %6, %9 : f64
        %11 = arith.addf %arg9, %10 : f64
        scf.yield %11 : f64
      }
      affine.store %5, %arg6[%arg7] : memref<?xf64>
    }
    return
  }
}

