module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_grad_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<5x4x4xf64>
    %alloca_1 = memref.alloca() : memref<5x4x4xf64>
    %alloca_2 = memref.alloca() : memref<5x5x4xf64>
    %alloca_3 = memref.alloca() : memref<5x5x4xf64>
    %alloca_4 = memref.alloca() : memref<5x5x4xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 4 {
            %0:3 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst, %arg10 = %cst, %arg11 = %cst) -> (f64, f64, f64) {
              %1 = affine.load %arg0[%arg4 * 375 + %arg5 + %arg8 * 25 + %arg6 * 5] : memref<?xf64>
              %2 = affine.load %arg2[%arg7 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg11, %3 : f64
              %5 = affine.load %arg0[%arg4 * 375 + %arg5 + %arg8 * 25 + %arg6 * 5 + 125] : memref<?xf64>
              %6 = affine.load %arg1[%arg7 + %arg8 * 4] : memref<?xf64>
              %7 = arith.mulf %5, %6 : f64
              %8 = arith.addf %arg10, %7 : f64
              %9 = affine.load %arg0[%arg4 * 375 + %arg5 + %arg8 * 25 + %arg6 * 5 + 250] : memref<?xf64>
              %10 = arith.mulf %9, %6 : f64
              %11 = arith.addf %arg9, %10 : f64
              affine.yield %11, %8, %4 : f64, f64, f64
            }
            affine.store %0#2, %alloca_4[%arg5, %arg6, %arg7] : memref<5x5x4xf64>
            affine.store %0#1, %alloca_3[%arg5, %arg6, %arg7] : memref<5x5x4xf64>
            affine.store %0#0, %alloca_2[%arg5, %arg6, %arg7] : memref<5x5x4xf64>
          }
        }
      }
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0:3 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst, %arg10 = %cst, %arg11 = %cst) -> (f64, f64, f64) {
              %1 = affine.load %alloca_4[%arg5, %arg8, %arg7] : memref<5x5x4xf64>
              %2 = affine.load %arg1[%arg6 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg11, %3 : f64
              %5 = affine.load %alloca_3[%arg5, %arg8, %arg7] : memref<5x5x4xf64>
              %6 = affine.load %arg2[%arg6 + %arg8 * 4] : memref<?xf64>
              %7 = arith.mulf %5, %6 : f64
              %8 = arith.addf %arg10, %7 : f64
              %9 = affine.load %alloca_2[%arg5, %arg8, %arg7] : memref<5x5x4xf64>
              %10 = arith.mulf %9, %2 : f64
              %11 = arith.addf %arg9, %10 : f64
              affine.yield %11, %8, %4 : f64, f64, f64
            }
            affine.store %0#2, %alloca_1[%arg5, %arg6, %arg7] : memref<5x4x4xf64>
            affine.store %0#1, %alloca_0[%arg5, %arg6, %arg7] : memref<5x4x4xf64>
            affine.store %0#0, %alloca[%arg5, %arg6, %arg7] : memref<5x4x4xf64>
          }
        }
      }
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0:3 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst, %arg10 = %cst, %arg11 = %cst) -> (f64, f64, f64) {
              %5 = affine.load %alloca_1[%arg8, %arg6, %arg7] : memref<5x4x4xf64>
              %6 = affine.load %arg1[%arg5 + %arg8 * 4] : memref<?xf64>
              %7 = arith.mulf %5, %6 : f64
              %8 = arith.addf %arg11, %7 : f64
              %9 = affine.load %alloca_0[%arg8, %arg6, %arg7] : memref<5x4x4xf64>
              %10 = arith.mulf %9, %6 : f64
              %11 = arith.addf %arg10, %10 : f64
              %12 = affine.load %alloca[%arg8, %arg6, %arg7] : memref<5x4x4xf64>
              %13 = affine.load %arg2[%arg5 + %arg8 * 4] : memref<?xf64>
              %14 = arith.mulf %12, %13 : f64
              %15 = arith.addf %arg9, %14 : f64
              affine.yield %15, %11, %8 : f64, f64, f64
            }
            %1 = arith.addf %0#2, %0#1 : f64
            %2 = arith.addf %1, %0#0 : f64
            %3 = affine.load %arg3[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg3[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
