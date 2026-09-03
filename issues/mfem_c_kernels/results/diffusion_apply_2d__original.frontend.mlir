module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x2xf64>
    %alloca_0 = memref.alloca() : memref<5x2xf64>
    %alloca_1 = memref.alloca() : memref<5x5x2xf64>
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.store %cst, %alloca_1[%arg8, %arg9, 1] : memref<5x5x2xf64>
          affine.store %cst, %alloca_1[%arg8, %arg9, 0] : memref<5x5x2xf64>
        }
      }
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.store %cst, %alloca_0[%arg9, 1] : memref<5x2xf64>
          affine.store %cst, %alloca_0[%arg9, 0] : memref<5x2xf64>
        }
        affine.for %arg9 = 0 to 4 {
          %0 = affine.load %arg5[%arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
          affine.for %arg10 = 0 to 5 {
            %1 = affine.load %arg0[%arg9 + %arg10 * 4] : memref<?xf64>
            %2 = arith.mulf %0, %1 : f64
            %3 = affine.load %alloca_0[%arg10, 0] : memref<5x2xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %alloca_0[%arg10, 0] : memref<5x2xf64>
            %5 = affine.load %arg1[%arg9 + %arg10 * 4] : memref<?xf64>
            %6 = arith.mulf %0, %5 : f64
            %7 = affine.load %alloca_0[%arg10, 1] : memref<5x2xf64>
            %8 = arith.addf %7, %6 : f64
            affine.store %8, %alloca_0[%arg10, 1] : memref<5x2xf64>
          }
        }
        affine.for %arg9 = 0 to 5 {
          %0 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
          %1 = affine.load %arg1[%arg8 + %arg9 * 4] : memref<?xf64>
          affine.for %arg10 = 0 to 5 {
            %2 = affine.load %alloca_0[%arg10, 1] : memref<5x2xf64>
            %3 = arith.mulf %2, %0 : f64
            %4 = affine.load %alloca_1[%arg9, %arg10, 0] : memref<5x5x2xf64>
            %5 = arith.addf %4, %3 : f64
            affine.store %5, %alloca_1[%arg9, %arg10, 0] : memref<5x5x2xf64>
            %6 = affine.load %alloca_0[%arg10, 0] : memref<5x2xf64>
            %7 = arith.mulf %6, %1 : f64
            %8 = affine.load %alloca_1[%arg9, %arg10, 1] : memref<5x5x2xf64>
            %9 = arith.addf %8, %7 : f64
            affine.store %9, %alloca_1[%arg9, %arg10, 1] : memref<5x5x2xf64>
          }
        }
      }
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          %0 = affine.load %alloca_1[%arg8, %arg9, 0] : memref<5x5x2xf64>
          %1 = affine.load %alloca_1[%arg8, %arg9, 1] : memref<5x5x2xf64>
          %2 = affine.load %arg4[%arg9 + %arg8 * 5 + %arg7 * 75] : memref<?xf64>
          %3 = affine.load %arg4[%arg9 + %arg7 * 75 + %arg8 * 5 + 25] : memref<?xf64>
          %4 = affine.load %arg4[%arg9 + %arg7 * 75 + %arg8 * 5 + 50] : memref<?xf64>
          %5 = arith.mulf %2, %0 : f64
          %6 = arith.mulf %3, %1 : f64
          %7 = arith.addf %5, %6 : f64
          affine.store %7, %alloca_1[%arg8, %arg9, 0] : memref<5x5x2xf64>
          %8 = arith.mulf %3, %0 : f64
          %9 = arith.mulf %4, %1 : f64
          %10 = arith.addf %8, %9 : f64
          affine.store %10, %alloca_1[%arg8, %arg9, 1] : memref<5x5x2xf64>
        }
      }
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.store %cst, %alloca[%arg9, 1] : memref<4x2xf64>
          affine.store %cst, %alloca[%arg9, 0] : memref<4x2xf64>
        }
        affine.for %arg9 = 0 to 5 {
          %0 = affine.load %alloca_1[%arg8, %arg9, 0] : memref<5x5x2xf64>
          %1 = affine.load %alloca_1[%arg8, %arg9, 1] : memref<5x5x2xf64>
          affine.for %arg10 = 0 to 4 {
            %2 = affine.load %arg3[%arg9 + %arg10 * 5] : memref<?xf64>
            %3 = arith.mulf %0, %2 : f64
            %4 = affine.load %alloca[%arg10, 0] : memref<4x2xf64>
            %5 = arith.addf %4, %3 : f64
            affine.store %5, %alloca[%arg10, 0] : memref<4x2xf64>
            %6 = affine.load %arg2[%arg9 + %arg10 * 5] : memref<?xf64>
            %7 = arith.mulf %1, %6 : f64
            %8 = affine.load %alloca[%arg10, 1] : memref<4x2xf64>
            %9 = arith.addf %8, %7 : f64
            affine.store %9, %alloca[%arg10, 1] : memref<4x2xf64>
          }
        }
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.load %alloca[%arg10, 0] : memref<4x2xf64>
            %1 = affine.load %arg2[%arg8 + %arg9 * 5] : memref<?xf64>
            %2 = arith.mulf %0, %1 : f64
            %3 = affine.load %alloca[%arg10, 1] : memref<4x2xf64>
            %4 = affine.load %arg3[%arg8 + %arg9 * 5] : memref<?xf64>
            %5 = arith.mulf %3, %4 : f64
            %6 = arith.addf %2, %5 : f64
            %7 = affine.load %arg6[%arg10 + %arg7 * 16 + %arg9 * 4] : memref<?xf64>
            %8 = arith.addf %7, %6 : f64
            affine.store %8, %arg6[%arg10 + %arg7 * 16 + %arg9 * 4] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
