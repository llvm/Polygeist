module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @openblas_dgemm_nt_b0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: f64, %arg6: i32, %arg7: memref<?x?xf64>, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    affine.for %arg10 = 0 to %0 {
      affine.for %arg11 = 0 to %1 {
        %3 = affine.for %arg12 = 0 to %2 iter_args(%arg13 = %cst) -> (f64) {
          %5 = affine.load %arg4[%arg12, %arg10] : memref<?x?xf64>
          %6 = affine.load %arg7[%arg12, %arg11] : memref<?x?xf64>
          %7 = arith.mulf %5, %6 : f64
          %8 = arith.addf %arg13, %7 : f64
          affine.yield %8 : f64
        }
        %4 = arith.mulf %arg5, %3 : f64
        affine.store %4, %arg9[%arg11, %arg10] : memref<?x?xf64>
      }
    }
    return
  }
}
