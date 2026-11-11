#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dpotf2_l(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant -1.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to %0 {
      %1 = affine.for %arg4 = 0 to #map(%arg3) iter_args(%arg5 = %cst) -> (f64) {
        %6 = affine.load %arg2[%arg4, %arg3] : memref<?x?xf64>
        %7 = arith.mulf %6, %6 : f64
        %8 = arith.addf %arg5, %7 : f64
        affine.yield %8 : f64
      }
      %2 = affine.load %arg2[%arg3, %arg3] : memref<?x?xf64>
      %3 = arith.subf %2, %1 : f64
      %4 = math.sqrt %3 : f64
      affine.store %4, %arg2[%arg3, %arg3] : memref<?x?xf64>
      affine.for %arg4 = 0 to #map(%arg3) {
        %6 = affine.load %arg2[%arg4, %arg3] : memref<?x?xf64>
        %7 = arith.mulf %6, %cst_0 : f64
        affine.for %arg5 = 0 to #map1(%arg3)[%0] {
          %8 = affine.load %arg2[%arg3, %arg3 + %arg5 + 1] : memref<?x?xf64>
          %9 = affine.load %arg2[%arg4, %arg5 + %arg3 + 1] : memref<?x?xf64>
          %10 = arith.mulf %7, %9 : f64
          %11 = arith.addf %8, %10 : f64
          affine.store %11, %arg2[%arg3, %arg3 + %arg5 + 1] : memref<?x?xf64>
        }
      }
      %5 = arith.divf %cst_1, %4 : f64
      affine.for %arg4 = 0 to #map1(%arg3)[%0] {
        %6 = affine.load %arg2[%arg3, %arg3 + %arg4 + 1] : memref<?x?xf64>
        %7 = arith.mulf %5, %6 : f64
        affine.store %7, %arg2[%arg3, %arg3 + %arg4 + 1] : memref<?x?xf64>
      }
    }
    return
  }
}
