module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x5xf64>
    %alloca_2 = memref.alloca() : memref<2x5x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x5xf64>
    %alloca_4 = memref.alloca() : memref<2x4x5xf64>
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %arg4[%arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_4[%arg6, %arg7, %arg8] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %arg4[%arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_3[%arg6, %arg7, %arg8] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_3[%arg6, %arg9, %arg8] : memref<2x4x5xf64>
            %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_2[%arg6, %arg7, %arg8] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_4[%arg6, %arg9, %arg8] : memref<2x4x5xf64>
            %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_1[%arg6, %arg7, %arg8] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %arg3[%arg9 + %arg6 * 50 + %arg7 * 5] : memref<?xf64>
            %2 = affine.load %alloca_2[%arg6, %arg7, %arg9] : memref<2x5x5xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = affine.load %arg3[%arg9 + %arg6 * 50 + %arg7 * 5 + 25] : memref<?xf64>
            %5 = affine.load %alloca_1[%arg6, %arg7, %arg9] : memref<2x5x5xf64>
            %6 = arith.mulf %4, %5 : f64
            %7 = arith.addf %3, %6 : f64
            %8 = affine.load %arg2[%arg9 + %arg8 * 5] : memref<?xf64>
            %9 = arith.mulf %7, %8 : f64
            %10 = arith.addf %arg10, %9 : f64
            affine.yield %10 : f64
          }
          affine.store %0, %alloca_0[%arg6, %arg7, %arg8] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_0[%arg6, %arg9, %arg8] : memref<2x5x4xf64>
            %2 = affine.load %arg2[%arg9 + %arg7 * 5] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca[%arg6, %arg7, %arg8] : memref<2x4x4xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.load %alloca[%arg6, %arg7, %arg8] : memref<2x4x4xf64>
          %1 = affine.load %arg5[%arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          %2 = arith.addf %1, %0 : f64
          affine.store %2, %arg5[%arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
        }
      }
    }
    return
  }
}
