#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x40xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 1 to #map()[%1] {
        affine.for %arg5 = 1 to #map()[%1] {
          %2 = affine.load %arg2[%arg4 - 1, %arg5 - 1] : memref<?x40xf64>
          %3 = affine.load %arg2[%arg4 - 1, %arg5] : memref<?x40xf64>
          %4 = arith.addf %2, %3 : f64
          %5 = affine.load %arg2[%arg4 - 1, %arg5 + 1] : memref<?x40xf64>
          %6 = arith.addf %4, %5 : f64
          %7 = affine.load %arg2[%arg4, %arg5 - 1] : memref<?x40xf64>
          %8 = arith.addf %6, %7 : f64
          %9 = affine.load %arg2[%arg4, %arg5] : memref<?x40xf64>
          %10 = arith.addf %8, %9 : f64
          %11 = affine.load %arg2[%arg4, %arg5 + 1] : memref<?x40xf64>
          %12 = arith.addf %10, %11 : f64
          %13 = affine.load %arg2[%arg4 + 1, %arg5 - 1] : memref<?x40xf64>
          %14 = arith.addf %12, %13 : f64
          %15 = affine.load %arg2[%arg4 + 1, %arg5] : memref<?x40xf64>
          %16 = arith.addf %14, %15 : f64
          %17 = affine.load %arg2[%arg4 + 1, %arg5 + 1] : memref<?x40xf64>
          %18 = arith.addf %16, %17 : f64
          %19 = arith.divf %18, %cst : f64
          affine.store %19, %arg2[%arg4, %arg5] : memref<?x40xf64>
        }
      }
    }
    return
  }
}
