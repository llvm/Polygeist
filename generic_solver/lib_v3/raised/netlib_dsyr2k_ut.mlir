#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyr2k_ut(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg10 = 0 to %0 {
      affine.for %arg11 = 0 to #map(%arg10) {
        %2:2 = affine.for %arg12 = 0 to %1 iter_args(%arg13 = %cst, %arg14 = %cst) -> (f64, f64) {
          %9 = affine.load %arg4[%arg11, %arg12] : memref<?x?xf64>
          %10 = affine.load %arg6[%arg10, %arg12] : memref<?x?xf64>
          %11 = arith.mulf %9, %10 : f64
          %12 = arith.addf %arg14, %11 : f64
          %13 = affine.load %arg6[%arg11, %arg12] : memref<?x?xf64>
          %14 = affine.load %arg4[%arg10, %arg12] : memref<?x?xf64>
          %15 = arith.mulf %13, %14 : f64
          %16 = arith.addf %arg13, %15 : f64
          affine.yield %16, %12 : f64, f64
        }
        %3 = affine.load %arg9[%arg10, %arg11] : memref<?x?xf64>
        %4 = arith.mulf %arg7, %3 : f64
        %5 = arith.mulf %arg2, %2#1 : f64
        %6 = arith.addf %4, %5 : f64
        %7 = arith.mulf %arg2, %2#0 : f64
        %8 = arith.addf %6, %7 : f64
        affine.store %8, %arg9[%arg10, %arg11] : memref<?x?xf64>
      }
    }
    return
  }
}
