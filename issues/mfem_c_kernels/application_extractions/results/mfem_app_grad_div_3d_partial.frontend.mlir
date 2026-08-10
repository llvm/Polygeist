module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_grad_div_3d_partial(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_7 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_8 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_9 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_10 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_11 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_13 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg8 * 12 + %arg11 + %arg9 * 4 + %arg7 * 108] : memref<?xf64>
              %2 = affine.load %arg2[%arg11 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_13[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg9 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg8 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_7[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg8 * 12 + %arg11 + %arg9 * 3 + %arg7 * 108 + 36] : memref<?xf64>
              %2 = affine.load %arg0[%arg11 + %arg10 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_12[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg2[%arg11 + %arg9 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_9[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg8 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg8 * 9 + %arg11 + %arg9 * 3 + %arg7 * 108 + 72] : memref<?xf64>
              %2 = affine.load %arg0[%arg11 + %arg10 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg9 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg2[%arg11 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg7 * 125 + %arg11 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_7[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_6[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %4 = arith.addf %2, %3 : f64
              %5 = affine.load %alloca_5[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.addf %4, %5 : f64
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %arg3[%arg11 + %arg10 * 5] : memref<?xf64>
              %9 = arith.mulf %7, %8 : f64
              %10 = arith.addf %arg12, %9 : f64
              affine.yield %10 : f64
            }
            affine.store %0, %alloca_4[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg7 * 125 + %arg11 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_7[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_6[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %4 = arith.addf %2, %3 : f64
              %5 = affine.load %alloca_5[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.addf %4, %5 : f64
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %arg1[%arg11 + %arg10 * 5] : memref<?xf64>
              %9 = arith.mulf %7, %8 : f64
              %10 = arith.addf %arg12, %9 : f64
              affine.yield %10 : f64
            }
            affine.store %0, %alloca_3[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg7 * 125 + %arg11 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_7[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_6[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %4 = arith.addf %2, %3 : f64
              %5 = affine.load %alloca_5[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.addf %4, %5 : f64
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %arg1[%arg11 + %arg10 * 5] : memref<?xf64>
              %9 = arith.mulf %7, %8 : f64
              %10 = arith.addf %arg12, %9 : f64
              affine.yield %10 : f64
            }
            affine.store %0, %alloca_2[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_4[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg1[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg1[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %alloca_1[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %4 = affine.load %arg1[%arg11 + %arg8 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg12, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg6[%arg8 * 12 + %arg10 + %arg9 * 4 + %arg7 * 108] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 12 + %arg10 + %arg9 * 4 + %arg7 * 108] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %alloca_0[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %4 = affine.load %arg1[%arg11 + %arg8 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg12, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg6[%arg8 * 12 + %arg10 + %arg9 * 3 + %arg7 * 108 + 36] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 12 + %arg10 + %arg9 * 3 + %arg7 * 108 + 36] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %4 = affine.load %arg3[%arg11 + %arg8 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg12, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg6[%arg8 * 9 + %arg10 + %arg9 * 3 + %arg7 * 108 + 72] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 9 + %arg10 + %arg9 * 3 + %arg7 * 108 + 72] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_divdiv_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_7 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_8 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_9 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_10 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_11 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_13 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg8 * 12 + %arg11 + %arg9 * 4 + %arg7 * 108] : memref<?xf64>
              %2 = affine.load %arg2[%arg11 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_13[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg9 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg8 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_7[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg8 * 12 + %arg11 + %arg9 * 3 + %arg7 * 108 + 36] : memref<?xf64>
              %2 = affine.load %arg0[%arg11 + %arg10 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_12[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg2[%arg11 + %arg9 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_9[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg8 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg8 * 9 + %arg11 + %arg9 * 3 + %arg7 * 108 + 72] : memref<?xf64>
              %2 = affine.load %arg0[%arg11 + %arg10 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg9 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg2[%arg11 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg7 * 125 + %arg11 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_7[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_6[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %4 = arith.addf %2, %3 : f64
              %5 = affine.load %alloca_5[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.addf %4, %5 : f64
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %arg3[%arg11 + %arg10 * 5] : memref<?xf64>
              %9 = arith.mulf %7, %8 : f64
              %10 = arith.addf %arg12, %9 : f64
              affine.yield %10 : f64
            }
            affine.store %0, %alloca_4[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg7 * 125 + %arg11 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_7[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_6[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %4 = arith.addf %2, %3 : f64
              %5 = affine.load %alloca_5[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.addf %4, %5 : f64
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %arg1[%arg11 + %arg10 * 5] : memref<?xf64>
              %9 = arith.mulf %7, %8 : f64
              %10 = arith.addf %arg12, %9 : f64
              affine.yield %10 : f64
            }
            affine.store %0, %alloca_3[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg7 * 125 + %arg11 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
              %2 = affine.load %alloca_7[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_6[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %4 = arith.addf %2, %3 : f64
              %5 = affine.load %alloca_5[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.addf %4, %5 : f64
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %arg1[%arg11 + %arg10 * 5] : memref<?xf64>
              %9 = arith.mulf %7, %8 : f64
              %10 = arith.addf %arg12, %9 : f64
              affine.yield %10 : f64
            }
            affine.store %0, %alloca_2[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_4[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg1[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg1[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %alloca_1[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %4 = affine.load %arg1[%arg11 + %arg8 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg12, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg6[%arg8 * 12 + %arg10 + %arg9 * 4 + %arg7 * 108] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 12 + %arg10 + %arg9 * 4 + %arg7 * 108] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %alloca_0[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %4 = affine.load %arg1[%arg11 + %arg8 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg12, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg6[%arg8 * 12 + %arg10 + %arg9 * 3 + %arg7 * 108 + 36] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 12 + %arg10 + %arg9 * 3 + %arg7 * 108 + 36] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %4 = affine.load %arg3[%arg11 + %arg8 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg12, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg6[%arg8 * 9 + %arg10 + %arg9 * 3 + %arg7 * 108 + 72] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 9 + %arg10 + %arg9 * 3 + %arg7 * 108 + 72] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
