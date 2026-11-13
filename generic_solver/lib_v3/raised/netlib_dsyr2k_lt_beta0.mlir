#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyr2k_lt_beta0(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: i32, %arg8: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg9 = 0 to %0 {
      affine.for %arg10 = #map(%arg9) to %0 {
        %2:2 = affine.for %arg11 = 0 to %1 iter_args(%arg12 = %cst, %arg13 = %cst) -> (f64, f64) {
          %6 = affine.load %arg4[%arg10, %arg11] : memref<?x?xf64>
          %7 = affine.load %arg6[%arg9, %arg11] : memref<?x?xf64>
          %8 = arith.mulf %6, %7 : f64
          %9 = arith.addf %arg13, %8 : f64
          %10 = affine.load %arg6[%arg10, %arg11] : memref<?x?xf64>
          %11 = affine.load %arg4[%arg9, %arg11] : memref<?x?xf64>
          %12 = arith.mulf %10, %11 : f64
          %13 = arith.addf %arg12, %12 : f64
          affine.yield %13, %9 : f64, f64
        }
        %3 = arith.mulf %arg2, %2#1 : f64
        %4 = arith.mulf %arg2, %2#0 : f64
        %5 = arith.addf %3, %4 : f64
        affine.store %5, %arg8[%arg9, %arg10] : memref<?x?xf64>
      }
    }
    return
  }
}
