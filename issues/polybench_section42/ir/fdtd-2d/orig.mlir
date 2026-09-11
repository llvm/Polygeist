#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.69999999999999996 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      affine.for %arg8 = 0 to %1 {
        %3 = affine.load %arg6[%arg7] : memref<?xf64>
        affine.store %3, %arg4[0, %arg8] : memref<?x?xf64>
      }
      affine.for %arg8 = 1 to %0 {
        affine.for %arg9 = 0 to %1 {
          %3 = affine.load %arg4[%arg8, %arg9] : memref<?x?xf64>
          %4 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
          %5 = affine.load %arg5[%arg8 - 1, %arg9] : memref<?x?xf64>
          %6 = arith.subf %4, %5 : f64
          %7 = arith.mulf %6, %cst : f64
          %8 = arith.subf %3, %7 : f64
          affine.store %8, %arg4[%arg8, %arg9] : memref<?x?xf64>
        }
      }
      affine.for %arg8 = 0 to %0 {
        affine.for %arg9 = 1 to %1 {
          %3 = affine.load %arg3[%arg8, %arg9] : memref<?x?xf64>
          %4 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
          %5 = affine.load %arg5[%arg8, %arg9 - 1] : memref<?x?xf64>
          %6 = arith.subf %4, %5 : f64
          %7 = arith.mulf %6, %cst : f64
          %8 = arith.subf %3, %7 : f64
          affine.store %8, %arg3[%arg8, %arg9] : memref<?x?xf64>
        }
      }
      affine.for %arg8 = 0 to #map()[%0] {
        affine.for %arg9 = 0 to #map()[%1] {
          %3 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
          %4 = affine.load %arg3[%arg8, %arg9 + 1] : memref<?x?xf64>
          %5 = affine.load %arg3[%arg8, %arg9] : memref<?x?xf64>
          %6 = arith.subf %4, %5 : f64
          %7 = affine.load %arg4[%arg8 + 1, %arg9] : memref<?x?xf64>
          %8 = arith.addf %6, %7 : f64
          %9 = affine.load %arg4[%arg8, %arg9] : memref<?x?xf64>
          %10 = arith.subf %8, %9 : f64
          %11 = arith.mulf %10, %cst_0 : f64
          %12 = arith.subf %3, %11 : f64
          affine.store %12, %arg5[%arg8, %arg9] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
