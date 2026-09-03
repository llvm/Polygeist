module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex9p_mass_convection_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: f64, %arg8: f64, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>, %arg12: memref<?xf64>, %arg13: memref<?xf64>, %arg14: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x5xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5xf64>
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 4 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %arg0[%arg18 + %arg17 * 4] : memref<?xf64>
            %3 = affine.load %arg5[%arg18 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_1[%arg15, %arg16, %arg17] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 5 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %arg0[%arg18 + %arg16 * 4] : memref<?xf64>
            %3 = affine.load %alloca_1[%arg15, %arg18, %arg17] : memref<2x4x5xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_0[%arg15, %arg16, %arg17] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 5 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.load %arg3[%arg17 + %arg15 * 25 + %arg16 * 5] : memref<?xf64>
          %2 = affine.load %alloca_0[%arg15, %arg16, %arg17] : memref<2x5x5xf64>
          %3 = arith.mulf %2, %1 : f64
          affine.store %3, %alloca_0[%arg15, %arg16, %arg17] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 5 {
        affine.for %arg17 = 0 to 4 {
          %1 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %arg2[%arg18 + %arg17 * 5] : memref<?xf64>
            %3 = affine.load %alloca_0[%arg15, %arg16, %arg18] : memref<2x5x5xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca[%arg15, %arg16, %arg17] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 4 {
        affine.for %arg17 = 0 to 4 {
          %1 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %cst) -> (f64) {
            %4 = affine.load %arg2[%arg18 + %arg16 * 5] : memref<?xf64>
            %5 = affine.load %alloca[%arg15, %arg18, %arg17] : memref<2x5x4xf64>
            %6 = arith.mulf %4, %5 : f64
            %7 = arith.addf %arg19, %6 : f64
            affine.yield %7 : f64
          }
          %2 = affine.load %arg9[%arg17 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
          %3 = arith.addf %2, %1 : f64
          affine.store %3, %arg9[%arg17 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
        }
      }
    }
    %alloca_2 = memref.alloca() : memref<2x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x4x5xf64>
    %alloca_7 = memref.alloca() : memref<2x4x5xf64>
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 4 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %arg5[%arg18 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
            %3 = affine.load %arg0[%arg18 + %arg17 * 4] : memref<?xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_7[%arg15, %arg16, %arg17] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 4 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %arg5[%arg18 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
            %3 = affine.load %arg1[%arg18 + %arg17 * 4] : memref<?xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_6[%arg15, %arg16, %arg17] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 5 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %alloca_6[%arg15, %arg18, %arg17] : memref<2x4x5xf64>
            %3 = affine.load %arg0[%arg18 + %arg16 * 4] : memref<?xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_5[%arg15, %arg16, %arg17] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 5 {
        affine.for %arg17 = 0 to 5 {
          %1 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %alloca_7[%arg15, %arg18, %arg17] : memref<2x4x5xf64>
            %3 = affine.load %arg1[%arg18 + %arg16 * 4] : memref<?xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_4[%arg15, %arg16, %arg17] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 5 {
        affine.for %arg17 = 0 to 4 {
          %1 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %arg4[%arg18 + %arg15 * 50 + %arg16 * 5] : memref<?xf64>
            %3 = affine.load %alloca_5[%arg15, %arg16, %arg18] : memref<2x5x5xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = affine.load %arg4[%arg18 + %arg15 * 50 + %arg16 * 5 + 25] : memref<?xf64>
            %6 = affine.load %alloca_4[%arg15, %arg16, %arg18] : memref<2x5x5xf64>
            %7 = arith.mulf %5, %6 : f64
            %8 = arith.addf %4, %7 : f64
            %9 = affine.load %arg2[%arg18 + %arg17 * 5] : memref<?xf64>
            %10 = arith.mulf %8, %9 : f64
            %11 = arith.addf %arg19, %10 : f64
            affine.yield %11 : f64
          }
          affine.store %1, %alloca_3[%arg15, %arg16, %arg17] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 4 {
        affine.for %arg17 = 0 to 4 {
          %1 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %cst) -> (f64) {
            %2 = affine.load %alloca_3[%arg15, %arg18, %arg17] : memref<2x5x4xf64>
            %3 = affine.load %arg2[%arg18 + %arg16 * 5] : memref<?xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %arg19, %4 : f64
            affine.yield %5 : f64
          }
          affine.store %1, %alloca_2[%arg15, %arg16, %arg17] : memref<2x4x4xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 2 {
      affine.for %arg16 = 0 to 4 {
        affine.for %arg17 = 0 to 4 {
          %1 = affine.load %alloca_2[%arg15, %arg16, %arg17] : memref<2x4x4xf64>
          %2 = affine.load %arg10[%arg17 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
          %3 = arith.addf %2, %1 : f64
          affine.store %3, %arg10[%arg17 + %arg15 * 16 + %arg16 * 4] : memref<?xf64>
        }
      }
    }
    affine.for %arg15 = 0 to 32 {
      %1 = affine.load %arg13[%arg15] : memref<?xf64>
      %2 = arith.mulf %arg7, %1 : f64
      %3 = affine.load %arg10[%arg15] : memref<?xf64>
      %4 = arith.addf %3, %2 : f64
      affine.store %4, %arg10[%arg15] : memref<?xf64>
    }
    affine.for %arg15 = 0 to 32 {
      %1 = affine.load %arg9[%arg15] : memref<?xf64>
      %2 = arith.mulf %arg7, %1 : f64
      %3 = affine.load %arg11[%arg15] : memref<?xf64>
      %4 = arith.subf %3, %2 : f64
      affine.store %4, %arg11[%arg15] : memref<?xf64>
    }
    affine.for %arg15 = 0 to 32 {
      %1 = affine.load %arg6[%arg15] : memref<?xf64>
      %2 = affine.load %arg11[%arg15] : memref<?xf64>
      %3 = arith.mulf %1, %2 : f64
      affine.store %3, %arg12[%arg15] : memref<?xf64>
    }
    %0 = affine.for %arg15 = 0 to 32 iter_args(%arg16 = %cst) -> (f64) {
      %1 = affine.load %arg11[%arg15] : memref<?xf64>
      %2 = affine.load %arg12[%arg15] : memref<?xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %arg16, %3 : f64
      affine.yield %4 : f64
    }
    affine.for %arg15 = 0 to 32 {
      %1 = affine.load %arg12[%arg15] : memref<?xf64>
      %2 = affine.load %arg13[%arg15] : memref<?xf64>
      %3 = arith.mulf %arg8, %2 : f64
      %4 = arith.addf %1, %3 : f64
      affine.store %4, %arg13[%arg15] : memref<?xf64>
    }
    affine.store %0, %arg14[0] : memref<?xf64>
    return
  }
  func.func @mfem_pa_mass_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x5xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
            %1 = affine.load %arg0[%arg8 + %arg7 * 4] : memref<?xf64>
            %2 = affine.load %arg3[%arg8 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg9, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_1[%arg5, %arg6, %arg7] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
            %1 = affine.load %arg0[%arg8 + %arg6 * 4] : memref<?xf64>
            %2 = affine.load %alloca_1[%arg5, %arg8, %arg7] : memref<2x4x5xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg9, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_0[%arg5, %arg6, %arg7] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          %0 = affine.load %arg2[%arg7 + %arg5 * 25 + %arg6 * 5] : memref<?xf64>
          %1 = affine.load %alloca_0[%arg5, %arg6, %arg7] : memref<2x5x5xf64>
          %2 = arith.mulf %1, %0 : f64
          affine.store %2, %alloca_0[%arg5, %arg6, %arg7] : memref<2x5x5xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
            %1 = affine.load %arg1[%arg8 + %arg7 * 5] : memref<?xf64>
            %2 = affine.load %alloca_0[%arg5, %arg6, %arg8] : memref<2x5x5xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg9, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca[%arg5, %arg6, %arg7] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
            %3 = affine.load %arg1[%arg8 + %arg6 * 5] : memref<?xf64>
            %4 = affine.load %alloca[%arg5, %arg8, %arg7] : memref<2x5x4xf64>
            %5 = arith.mulf %3, %4 : f64
            %6 = arith.addf %arg9, %5 : f64
            affine.yield %6 : f64
          }
          %1 = affine.load %arg4[%arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
          %2 = arith.addf %1, %0 : f64
          affine.store %2, %arg4[%arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
        }
      }
    }
    return
  }
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
  func.func @mfem_mass_pcg_step_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: f64, %arg3: f64, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg9 = 0 to 32 {
      %1 = affine.load %arg7[%arg9] : memref<?xf64>
      %2 = arith.mulf %arg2, %1 : f64
      %3 = affine.load %arg4[%arg9] : memref<?xf64>
      %4 = arith.addf %3, %2 : f64
      affine.store %4, %arg4[%arg9] : memref<?xf64>
    }
    affine.for %arg9 = 0 to 32 {
      %1 = affine.load %arg0[%arg9] : memref<?xf64>
      %2 = arith.mulf %arg2, %1 : f64
      %3 = affine.load %arg5[%arg9] : memref<?xf64>
      %4 = arith.subf %3, %2 : f64
      affine.store %4, %arg5[%arg9] : memref<?xf64>
    }
    affine.for %arg9 = 0 to 32 {
      %1 = affine.load %arg1[%arg9] : memref<?xf64>
      %2 = affine.load %arg5[%arg9] : memref<?xf64>
      %3 = arith.mulf %1, %2 : f64
      affine.store %3, %arg6[%arg9] : memref<?xf64>
    }
    %0 = affine.for %arg9 = 0 to 32 iter_args(%arg10 = %cst) -> (f64) {
      %1 = affine.load %arg5[%arg9] : memref<?xf64>
      %2 = affine.load %arg6[%arg9] : memref<?xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %arg10, %3 : f64
      affine.yield %4 : f64
    }
    affine.for %arg9 = 0 to 32 {
      %1 = affine.load %arg6[%arg9] : memref<?xf64>
      %2 = affine.load %arg7[%arg9] : memref<?xf64>
      %3 = arith.mulf %arg3, %2 : f64
      %4 = arith.addf %1, %3 : f64
      affine.store %4, %arg7[%arg9] : memref<?xf64>
    }
    affine.store %0, %arg8[0] : memref<?xf64>
    return
  }
}
