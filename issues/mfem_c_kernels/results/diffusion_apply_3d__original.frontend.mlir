module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x3xf64>
    %alloca_0 = memref.alloca() : memref<4x4x3xf64>
    %alloca_1 = memref.alloca() : memref<5x2xf64>
    %alloca_2 = memref.alloca() : memref<5x5x3xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5x3xf64>
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            affine.for %arg11 = 0 to 3 {
              affine.store %cst, %alloca_3[%arg8, %arg9, %arg10, %arg11] : memref<5x5x5x3xf64>
            }
          }
        }
      }
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            affine.for %arg11 = 0 to 3 {
              affine.store %cst, %alloca_2[%arg9, %arg10, %arg11] : memref<5x5x3xf64>
            }
          }
        }
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 5 {
            affine.store %cst, %alloca_1[%arg10, 1] : memref<5x2xf64>
            affine.store %cst, %alloca_1[%arg10, 0] : memref<5x2xf64>
          }
          affine.for %arg10 = 0 to 4 {
            %0 = affine.load %arg5[%arg7 * 64 + %arg10 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
            affine.for %arg11 = 0 to 5 {
              %1 = affine.load %arg0[%arg10 + %arg11 * 4] : memref<?xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_1[%arg11, 0] : memref<5x2xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_1[%arg11, 0] : memref<5x2xf64>
              %5 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
              %6 = arith.mulf %0, %5 : f64
              %7 = affine.load %alloca_1[%arg11, 1] : memref<5x2xf64>
              %8 = arith.addf %7, %6 : f64
              affine.store %8, %alloca_1[%arg11, 1] : memref<5x2xf64>
            }
          }
          affine.for %arg10 = 0 to 5 {
            %0 = affine.load %arg0[%arg9 + %arg10 * 4] : memref<?xf64>
            %1 = affine.load %arg1[%arg9 + %arg10 * 4] : memref<?xf64>
            affine.for %arg11 = 0 to 5 {
              %2 = affine.load %alloca_1[%arg11, 1] : memref<5x2xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_2[%arg10, %arg11, 0] : memref<5x5x3xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_2[%arg10, %arg11, 0] : memref<5x5x3xf64>
              %6 = affine.load %alloca_1[%arg11, 0] : memref<5x2xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_2[%arg10, %arg11, 1] : memref<5x5x3xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_2[%arg10, %arg11, 1] : memref<5x5x3xf64>
              %10 = arith.mulf %6, %0 : f64
              %11 = affine.load %alloca_2[%arg10, %arg11, 2] : memref<5x5x3xf64>
              %12 = arith.addf %11, %10 : f64
              affine.store %12, %alloca_2[%arg10, %arg11, 2] : memref<5x5x3xf64>
            }
          }
        }
        affine.for %arg9 = 0 to 5 {
          %0 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
          %1 = affine.load %arg1[%arg8 + %arg9 * 4] : memref<?xf64>
          affine.for %arg10 = 0 to 5 {
            affine.for %arg11 = 0 to 5 {
              %2 = affine.load %alloca_2[%arg10, %arg11, 0] : memref<5x5x3xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_3[%arg9, %arg10, %arg11, 0] : memref<5x5x5x3xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_3[%arg9, %arg10, %arg11, 0] : memref<5x5x5x3xf64>
              %6 = affine.load %alloca_2[%arg10, %arg11, 1] : memref<5x5x3xf64>
              %7 = arith.mulf %6, %0 : f64
              %8 = affine.load %alloca_3[%arg9, %arg10, %arg11, 1] : memref<5x5x5x3xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_3[%arg9, %arg10, %arg11, 1] : memref<5x5x5x3xf64>
              %10 = affine.load %alloca_2[%arg10, %arg11, 2] : memref<5x5x3xf64>
              %11 = arith.mulf %10, %1 : f64
              %12 = affine.load %alloca_3[%arg9, %arg10, %arg11, 2] : memref<5x5x5x3xf64>
              %13 = arith.addf %12, %11 : f64
              affine.store %13, %alloca_3[%arg9, %arg10, %arg11, 2] : memref<5x5x5x3xf64>
            }
          }
        }
      }
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.load %alloca_3[%arg8, %arg9, %arg10, 0] : memref<5x5x5x3xf64>
            %1 = affine.load %alloca_3[%arg8, %arg9, %arg10, 1] : memref<5x5x5x3xf64>
            %2 = affine.load %alloca_3[%arg8, %arg9, %arg10, 2] : memref<5x5x5x3xf64>
            %3 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750] : memref<?xf64>
            %4 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg7 * 750 + %arg9 * 5 + 125] : memref<?xf64>
            %5 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg7 * 750 + %arg9 * 5 + 250] : memref<?xf64>
            %6 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg7 * 750 + %arg9 * 5 + 375] : memref<?xf64>
            %7 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg7 * 750 + %arg9 * 5 + 500] : memref<?xf64>
            %8 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg7 * 750 + %arg9 * 5 + 625] : memref<?xf64>
            %9 = arith.mulf %3, %0 : f64
            %10 = arith.mulf %4, %1 : f64
            %11 = arith.addf %9, %10 : f64
            %12 = arith.mulf %5, %2 : f64
            %13 = arith.addf %11, %12 : f64
            affine.store %13, %alloca_3[%arg8, %arg9, %arg10, 0] : memref<5x5x5x3xf64>
            %14 = arith.mulf %4, %0 : f64
            %15 = arith.mulf %6, %1 : f64
            %16 = arith.addf %14, %15 : f64
            %17 = arith.mulf %7, %2 : f64
            %18 = arith.addf %16, %17 : f64
            affine.store %18, %alloca_3[%arg8, %arg9, %arg10, 1] : memref<5x5x5x3xf64>
            %19 = arith.mulf %5, %0 : f64
            %20 = arith.mulf %7, %1 : f64
            %21 = arith.addf %19, %20 : f64
            %22 = arith.mulf %8, %2 : f64
            %23 = arith.addf %21, %22 : f64
            affine.store %23, %alloca_3[%arg8, %arg9, %arg10, 2] : memref<5x5x5x3xf64>
          }
        }
      }
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            affine.for %arg11 = 0 to 3 {
              affine.store %cst, %alloca_0[%arg9, %arg10, %arg11] : memref<4x4x3xf64>
            }
          }
        }
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 4 {
            affine.for %arg11 = 0 to 3 {
              affine.store %cst, %alloca[%arg10, %arg11] : memref<4x3xf64>
            }
          }
          affine.for %arg10 = 0 to 5 {
            %0 = affine.load %alloca_3[%arg8, %arg9, %arg10, 0] : memref<5x5x5x3xf64>
            %1 = affine.load %alloca_3[%arg8, %arg9, %arg10, 1] : memref<5x5x5x3xf64>
            %2 = affine.load %alloca_3[%arg8, %arg9, %arg10, 2] : memref<5x5x5x3xf64>
            affine.for %arg11 = 0 to 4 {
              %3 = affine.load %arg3[%arg10 + %arg11 * 5] : memref<?xf64>
              %4 = arith.mulf %0, %3 : f64
              %5 = affine.load %alloca[%arg11, 0] : memref<4x3xf64>
              %6 = arith.addf %5, %4 : f64
              affine.store %6, %alloca[%arg11, 0] : memref<4x3xf64>
              %7 = affine.load %arg2[%arg10 + %arg11 * 5] : memref<?xf64>
              %8 = arith.mulf %1, %7 : f64
              %9 = affine.load %alloca[%arg11, 1] : memref<4x3xf64>
              %10 = arith.addf %9, %8 : f64
              affine.store %10, %alloca[%arg11, 1] : memref<4x3xf64>
              %11 = affine.load %arg2[%arg10 + %arg11 * 5] : memref<?xf64>
              %12 = arith.mulf %2, %11 : f64
              %13 = affine.load %alloca[%arg11, 2] : memref<4x3xf64>
              %14 = arith.addf %13, %12 : f64
              affine.store %14, %alloca[%arg11, 2] : memref<4x3xf64>
            }
          }
          affine.for %arg10 = 0 to 4 {
            %0 = affine.load %arg2[%arg9 + %arg10 * 5] : memref<?xf64>
            %1 = affine.load %arg3[%arg9 + %arg10 * 5] : memref<?xf64>
            affine.for %arg11 = 0 to 4 {
              %2 = affine.load %alloca[%arg11, 0] : memref<4x3xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_0[%arg10, %arg11, 0] : memref<4x4x3xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_0[%arg10, %arg11, 0] : memref<4x4x3xf64>
              %6 = affine.load %alloca[%arg11, 1] : memref<4x3xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_0[%arg10, %arg11, 1] : memref<4x4x3xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_0[%arg10, %arg11, 1] : memref<4x4x3xf64>
              %10 = affine.load %alloca[%arg11, 2] : memref<4x3xf64>
              %11 = arith.mulf %10, %0 : f64
              %12 = affine.load %alloca_0[%arg10, %arg11, 2] : memref<4x4x3xf64>
              %13 = arith.addf %12, %11 : f64
              affine.store %13, %alloca_0[%arg10, %arg11, 2] : memref<4x4x3xf64>
            }
          }
        }
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            affine.for %arg11 = 0 to 4 {
              %0 = affine.load %alloca_0[%arg10, %arg11, 0] : memref<4x4x3xf64>
              %1 = affine.load %alloca_0[%arg10, %arg11, 1] : memref<4x4x3xf64>
              %2 = arith.addf %0, %1 : f64
              %3 = affine.load %arg2[%arg8 + %arg9 * 5] : memref<?xf64>
              %4 = arith.mulf %2, %3 : f64
              %5 = affine.load %alloca_0[%arg10, %arg11, 2] : memref<4x4x3xf64>
              %6 = affine.load %arg3[%arg8 + %arg9 * 5] : memref<?xf64>
              %7 = arith.mulf %5, %6 : f64
              %8 = arith.addf %4, %7 : f64
              %9 = affine.load %arg6[%arg7 * 64 + %arg11 + %arg9 * 16 + %arg10 * 4] : memref<?xf64>
              %10 = arith.addf %9, %8 : f64
              affine.store %10, %arg6[%arg7 * 64 + %arg11 + %arg9 * 16 + %arg10 * 4] : memref<?xf64>
            }
          }
        }
      }
    }
    return
  }
}
