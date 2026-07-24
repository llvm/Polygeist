module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x4x5xf64>
    %alloca_0 = memref.alloca() : memref<4x5x5xf64>
    %alloca_1 = memref.alloca() : memref<5x5x5xf64>
    %alloca_2 = memref.alloca() : memref<5x5x5xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5xf64>
    %alloca_4 = memref.alloca() : memref<5x5x5xf64>
    %alloca_5 = memref.alloca() : memref<4x5x5xf64>
    %alloca_6 = memref.alloca() : memref<4x5x5xf64>
    %alloca_7 = memref.alloca() : memref<4x5x5xf64>
    %alloca_8 = memref.alloca() : memref<4x4x5xf64>
    %alloca_9 = memref.alloca() : memref<4x4x5xf64>
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          affine.for %arg9 = 0 to 5 {
            affine.store %cst, %alloca_8[%arg7, %arg8, %arg9] : memref<4x4x5xf64>
            affine.store %cst, %alloca_9[%arg7, %arg8, %arg9] : memref<4x4x5xf64>
            %0:2 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst, %arg12 = %cst) -> (f64, f64) {
              %1 = affine.load %arg0[%arg10 + %arg9 * 4] : memref<?xf64>
              %2 = affine.load %arg4[%arg6 * 64 + %arg10 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.store %4, %alloca_9[%arg7, %arg8, %arg9] : memref<4x4x5xf64>
              %5 = affine.load %arg1[%arg10 + %arg9 * 4] : memref<?xf64>
              %6 = arith.mulf %5, %2 : f64
              %7 = arith.addf %arg11, %6 : f64
              affine.store %7, %alloca_8[%arg7, %arg8, %arg9] : memref<4x4x5xf64>
              affine.yield %7, %4 : f64, f64
            }
          }
        }
      }
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 5 {
            affine.store %cst, %alloca_5[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
            affine.store %cst, %alloca_6[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
            affine.store %cst, %alloca_7[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
            %0:3 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst) -> (f64, f64, f64) {
              %1 = affine.load %arg0[%arg10 + %arg9 * 4] : memref<?xf64>
              %2 = affine.load %alloca_9[%arg7, %arg10, %arg8] : memref<4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg13, %3 : f64
              affine.store %4, %alloca_7[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
              %5 = affine.load %arg1[%arg10 + %arg9 * 4] : memref<?xf64>
              %6 = arith.mulf %5, %2 : f64
              %7 = arith.addf %arg12, %6 : f64
              affine.store %7, %alloca_6[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
              %8 = affine.load %arg0[%arg10 + %arg9 * 4] : memref<?xf64>
              %9 = affine.load %alloca_8[%arg7, %arg10, %arg8] : memref<4x4x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %arg11, %10 : f64
              affine.store %11, %alloca_5[%arg7, %arg9, %arg8] : memref<4x5x5xf64>
              affine.yield %11, %7, %4 : f64, f64, f64
            }
          }
        }
      }
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 5 {
            affine.store %cst, %alloca_2[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            affine.store %cst, %alloca_3[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            affine.store %cst, %alloca_4[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %0:3 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst) -> (f64, f64, f64) {
              %12 = affine.load %arg1[%arg10 + %arg9 * 4] : memref<?xf64>
              %13 = affine.load %alloca_7[%arg10, %arg8, %arg7] : memref<4x5x5xf64>
              %14 = arith.mulf %12, %13 : f64
              %15 = arith.addf %arg13, %14 : f64
              affine.store %15, %alloca_4[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
              %16 = affine.load %arg0[%arg10 + %arg9 * 4] : memref<?xf64>
              %17 = affine.load %alloca_6[%arg10, %arg8, %arg7] : memref<4x5x5xf64>
              %18 = arith.mulf %16, %17 : f64
              %19 = arith.addf %arg12, %18 : f64
              affine.store %19, %alloca_3[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
              %20 = affine.load %arg0[%arg10 + %arg9 * 4] : memref<?xf64>
              %21 = affine.load %alloca_5[%arg10, %arg8, %arg7] : memref<4x5x5xf64>
              %22 = arith.mulf %20, %21 : f64
              %23 = arith.addf %arg11, %22 : f64
              affine.store %23, %alloca_2[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
              affine.yield %23, %19, %15 : f64, f64, f64
            }
            %1 = affine.load %arg3[%arg6 * 375 + %arg7 + %arg9 * 25 + %arg8 * 5] : memref<?xf64>
            %2 = affine.load %alloca_2[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = affine.load %arg3[%arg6 * 375 + %arg7 + %arg9 * 25 + %arg8 * 5 + 125] : memref<?xf64>
            %5 = affine.load %alloca_3[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %6 = arith.mulf %4, %5 : f64
            %7 = arith.addf %3, %6 : f64
            %8 = affine.load %arg3[%arg6 * 375 + %arg7 + %arg9 * 25 + %arg8 * 5 + 250] : memref<?xf64>
            %9 = affine.load %alloca_4[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
            %10 = arith.mulf %8, %9 : f64
            %11 = arith.addf %7, %10 : f64
            affine.store %11, %alloca_1[%arg9, %arg8, %arg7] : memref<5x5x5xf64>
          }
        }
      }
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 4 {
            affine.store %cst, %alloca_0[%arg9, %arg8, %arg7] : memref<4x5x5xf64>
            %0 = affine.for %arg10 = 0 to 5 iter_args(%arg11 = %cst) -> (f64) {
              %1 = affine.load %arg2[%arg10 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_1[%arg10, %arg8, %arg7] : memref<5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg11, %3 : f64
              affine.store %4, %alloca_0[%arg9, %arg8, %arg7] : memref<4x5x5xf64>
              affine.yield %4 : f64
            }
          }
        }
      }
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          affine.for %arg9 = 0 to 4 {
            affine.store %cst, %alloca[%arg7, %arg9, %arg8] : memref<4x4x5xf64>
            %0 = affine.for %arg10 = 0 to 5 iter_args(%arg11 = %cst) -> (f64) {
              %1 = affine.load %arg2[%arg10 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_0[%arg7, %arg10, %arg8] : memref<4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg11, %3 : f64
              affine.store %4, %alloca[%arg7, %arg9, %arg8] : memref<4x4x5xf64>
              affine.yield %4 : f64
            }
          }
        }
      }
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 5 {
              %0 = affine.load %arg2[%arg10 + %arg9 * 5] : memref<?xf64>
              %1 = affine.load %alloca[%arg7, %arg8, %arg10] : memref<4x4x5xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %arg5[%arg6 * 64 + %arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %arg5[%arg6 * 64 + %arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
            }
          }
        }
      }
    }
    return
  }
}
