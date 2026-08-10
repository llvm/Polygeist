module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_curlcurl_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_4 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_9 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_13 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_14 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_15 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_16 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_17 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_18 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_21 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_22 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_23 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_24 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_25 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_26 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_27 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_28 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_29 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_30 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_31 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_32 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_33 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg8[%arg11 * 12 + %arg14 + %arg12 * 3 + %arg10 * 144] : memref<?xf64>
              %2 = affine.load %arg0[%arg14 + %arg13 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_33[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_33[%arg10, %arg11, %arg14, %arg13] : memref<2x4x4x5xf64>
              %2 = affine.load %arg4[%arg14 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_28[%arg10, %arg11, %arg12, %arg13] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_33[%arg10, %arg11, %arg14, %arg13] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg14 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_27[%arg10, %arg11, %arg12, %arg13] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_28[%arg10, %arg14, %arg12, %arg13] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg14 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_22[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_27[%arg10, %arg14, %arg12, %arg13] : memref<2x4x5x5xf64>
              %2 = affine.load %arg4[%arg14 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_21[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg8[%arg11 * 12 + %arg14 + %arg12 * 4 + %arg10 * 144 + 48] : memref<?xf64>
              %2 = affine.load %arg4[%arg14 + %arg13 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_32[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg8[%arg11 * 12 + %arg14 + %arg12 * 4 + %arg10 * 144 + 48] : memref<?xf64>
              %2 = affine.load %arg1[%arg14 + %arg13 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_31[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_32[%arg10, %arg11, %arg14, %arg13] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg14 + %arg12 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_26[%arg10, %arg11, %arg12, %arg13] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_31[%arg10, %arg11, %arg14, %arg13] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg14 + %arg12 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_25[%arg10, %arg11, %arg12, %arg13] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_26[%arg10, %arg14, %arg12, %arg13] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg14 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg10, %arg14, %arg12, %arg13] : memref<2x4x5x5xf64>
              %2 = affine.load %arg4[%arg14 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg8[%arg11 * 16 + %arg14 + %arg12 * 4 + %arg10 * 144 + 96] : memref<?xf64>
              %2 = affine.load %arg4[%arg14 + %arg13 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_30[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg8[%arg11 * 16 + %arg14 + %arg12 * 4 + %arg10 * 144 + 96] : memref<?xf64>
              %2 = affine.load %arg1[%arg14 + %arg13 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_29[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_30[%arg10, %arg11, %arg14, %arg13] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg14 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_24[%arg10, %arg11, %arg12, %arg13] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_29[%arg10, %arg11, %arg14, %arg13] : memref<2x4x4x5xf64>
              %2 = affine.load %arg4[%arg14 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_23[%arg10, %arg11, %arg12, %arg13] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_24[%arg10, %arg14, %arg12, %arg13] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg14 + %arg11 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_18[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_23[%arg10, %arg14, %arg12, %arg13] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg14 + %arg11 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_17[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 125] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 375] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 500] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg2[%arg14 + %arg13 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg15, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_16[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_16[%arg10, %arg11, %arg14, %arg13] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg14 + %arg12 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg10, %arg11, %arg12, %arg13] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg10, %arg14, %arg12, %arg13] : memref<2x5x4x4xf64>
              %2 = affine.load %arg5[%arg14 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 250] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 500] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 625] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg2[%arg14 + %arg13 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg15, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_15[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg10, %arg11, %arg14, %arg13] : memref<2x5x5x4xf64>
              %2 = affine.load %arg5[%arg14 + %arg12 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg10, %arg11, %arg12, %arg13] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_9[%arg10, %arg14, %arg12, %arg13] : memref<2x5x4x4xf64>
              %2 = affine.load %arg3[%arg14 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 250] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 500] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 625] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg5[%arg14 + %arg13 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg15, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_14[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_14[%arg10, %arg11, %arg14, %arg13] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg14 + %arg12 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg10, %arg11, %arg12, %arg13] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg10, %arg14, %arg12, %arg13] : memref<2x5x4x4xf64>
              %2 = affine.load %arg3[%arg14 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 125] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 250] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg3[%arg14 + %arg13 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg15, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_13[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_13[%arg10, %arg11, %arg14, %arg13] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg14 + %arg12 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_7[%arg10, %arg11, %arg12, %arg13] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_7[%arg10, %arg14, %arg12, %arg13] : memref<2x5x4x4xf64>
              %2 = affine.load %arg5[%arg14 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 125] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 250] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg3[%arg14 + %arg13 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg15, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_12[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg10, %arg11, %arg14, %arg13] : memref<2x5x5x4xf64>
              %2 = affine.load %arg5[%arg14 + %arg12 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg10, %arg11, %arg12, %arg13] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_6[%arg10, %arg14, %arg12, %arg13] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg14 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 125] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 375] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg10 * 750 + %arg14 + %arg11 * 25 + %arg12 * 5 + 500] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg10, %arg11, %arg12, %arg14] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg5[%arg14 + %arg13 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg15, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_11[%arg10, %arg11, %arg12, %arg13] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg10, %arg11, %arg14, %arg13] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg14 + %arg12 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg10, %arg11, %arg12, %arg13] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg10, %arg14, %arg12, %arg13] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg14 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg15, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.load %alloca_4[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_3[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
            %2 = arith.subf %0, %1 : f64
            %3 = affine.load %arg9[%arg11 * 12 + %arg13 + %arg12 * 3 + %arg10 * 144] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg9[%arg11 * 12 + %arg13 + %arg12 * 3 + %arg10 * 144] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.load %alloca_2[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_1[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
            %2 = arith.subf %0, %1 : f64
            %3 = affine.load %arg9[%arg11 * 12 + %arg13 + %arg12 * 4 + %arg10 * 144 + 48] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg9[%arg11 * 12 + %arg13 + %arg12 * 4 + %arg10 * 144 + 48] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.load %alloca_0[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca[%arg10, %arg11, %arg12, %arg13] : memref<2x4x4x4xf64>
            %2 = arith.subf %0, %1 : f64
            %3 = affine.load %arg9[%arg11 * 16 + %arg13 + %arg12 * 4 + %arg10 * 144 + 96] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg9[%arg11 * 16 + %arg13 + %arg12 * 4 + %arg10 * 144 + 96] : memref<?xf64>
          }
        }
      }
    }
    %alloca_34 = memref.alloca() : memref<2x3x5x5x5xf64>
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg1[%arg14 + %arg11 * 4] : memref<?xf64>
              %2 = affine.for %arg16 = 0 to 4 iter_args(%arg17 = %arg15) -> (f64) {
                %3 = affine.load %arg1[%arg16 + %arg12 * 4] : memref<?xf64>
                %4 = affine.for %arg18 = 0 to 3 iter_args(%arg19 = %arg17) -> (f64) {
                  %5 = affine.load %arg8[%arg14 * 12 + %arg18 + %arg16 * 3 + %arg10 * 144] : memref<?xf64>
                  %6 = affine.load %arg0[%arg18 + %arg13 * 3] : memref<?xf64>
                  %7 = arith.mulf %5, %6 : f64
                  %8 = arith.mulf %7, %3 : f64
                  %9 = arith.mulf %8, %1 : f64
                  %10 = arith.addf %arg19, %9 : f64
                  affine.yield %10 : f64
                }
                affine.yield %4 : f64
              }
              affine.yield %2 : f64
            }
            affine.store %0, %alloca_34[%arg10, 0, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg1[%arg14 + %arg11 * 4] : memref<?xf64>
              %2 = affine.for %arg16 = 0 to 3 iter_args(%arg17 = %arg15) -> (f64) {
                %3 = affine.load %arg0[%arg16 + %arg12 * 3] : memref<?xf64>
                %4 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %arg17) -> (f64) {
                  %5 = affine.load %arg8[%arg14 * 12 + %arg18 + %arg16 * 4 + %arg10 * 144 + 48] : memref<?xf64>
                  %6 = affine.load %arg1[%arg18 + %arg13 * 4] : memref<?xf64>
                  %7 = arith.mulf %5, %6 : f64
                  %8 = arith.mulf %7, %3 : f64
                  %9 = arith.mulf %8, %1 : f64
                  %10 = arith.addf %arg19, %9 : f64
                  affine.yield %10 : f64
                }
                affine.yield %4 : f64
              }
              affine.yield %2 : f64
            }
            affine.store %0, %alloca_34[%arg10, 1, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg14 + %arg11 * 3] : memref<?xf64>
              %2 = affine.for %arg16 = 0 to 4 iter_args(%arg17 = %arg15) -> (f64) {
                %3 = affine.load %arg1[%arg16 + %arg12 * 4] : memref<?xf64>
                %4 = affine.for %arg18 = 0 to 4 iter_args(%arg19 = %arg17) -> (f64) {
                  %5 = affine.load %arg8[%arg14 * 16 + %arg18 + %arg16 * 4 + %arg10 * 144 + 96] : memref<?xf64>
                  %6 = affine.load %arg1[%arg18 + %arg13 * 4] : memref<?xf64>
                  %7 = arith.mulf %5, %6 : f64
                  %8 = arith.mulf %7, %3 : f64
                  %9 = arith.mulf %8, %1 : f64
                  %10 = arith.addf %arg19, %9 : f64
                  affine.yield %10 : f64
                }
                affine.yield %4 : f64
              }
              affine.yield %2 : f64
            }
            affine.store %0, %alloca_34[%arg10, 2, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 5 {
        affine.for %arg12 = 0 to 5 {
          affine.for %arg13 = 0 to 5 {
            %0 = affine.load %alloca_34[%arg10, 0, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
            %1 = affine.load %alloca_34[%arg10, 1, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
            %2 = affine.load %alloca_34[%arg10, 2, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
            %3 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750] : memref<?xf64>
            %4 = arith.mulf %3, %0 : f64
            %5 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 125] : memref<?xf64>
            %6 = arith.mulf %5, %1 : f64
            %7 = arith.addf %4, %6 : f64
            %8 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 250] : memref<?xf64>
            %9 = arith.mulf %8, %2 : f64
            %10 = arith.addf %7, %9 : f64
            affine.store %10, %alloca_34[%arg10, 0, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
            %11 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 125] : memref<?xf64>
            %12 = arith.mulf %11, %0 : f64
            %13 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 375] : memref<?xf64>
            %14 = arith.mulf %13, %1 : f64
            %15 = arith.addf %12, %14 : f64
            %16 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 500] : memref<?xf64>
            %17 = arith.mulf %16, %2 : f64
            %18 = arith.addf %15, %17 : f64
            affine.store %18, %alloca_34[%arg10, 1, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
            %19 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 250] : memref<?xf64>
            %20 = arith.mulf %19, %0 : f64
            %21 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 500] : memref<?xf64>
            %22 = arith.mulf %21, %1 : f64
            %23 = arith.addf %20, %22 : f64
            %24 = affine.load %arg7[%arg11 * 25 + %arg13 + %arg12 * 5 + %arg10 * 750 + 625] : memref<?xf64>
            %25 = arith.mulf %24, %2 : f64
            %26 = arith.addf %23, %25 : f64
            affine.store %26, %alloca_34[%arg10, 2, %arg11, %arg12, %arg13] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 3 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %3 = affine.load %arg3[%arg14 + %arg11 * 5] : memref<?xf64>
              %4 = affine.for %arg16 = 0 to 5 iter_args(%arg17 = %arg15) -> (f64) {
                %5 = affine.load %arg3[%arg16 + %arg12 * 5] : memref<?xf64>
                %6 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %arg17) -> (f64) {
                  %7 = affine.load %alloca_34[%arg10, 0, %arg14, %arg16, %arg18] : memref<2x3x5x5x5xf64>
                  %8 = affine.load %arg2[%arg18 + %arg13 * 5] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg19, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg9[%arg11 * 12 + %arg13 + %arg12 * 3 + %arg10 * 144] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg9[%arg11 * 12 + %arg13 + %arg12 * 3 + %arg10 * 144] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 3 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %3 = affine.load %arg3[%arg14 + %arg11 * 5] : memref<?xf64>
              %4 = affine.for %arg16 = 0 to 5 iter_args(%arg17 = %arg15) -> (f64) {
                %5 = affine.load %arg2[%arg16 + %arg12 * 5] : memref<?xf64>
                %6 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %arg17) -> (f64) {
                  %7 = affine.load %alloca_34[%arg10, 1, %arg14, %arg16, %arg18] : memref<2x3x5x5x5xf64>
                  %8 = affine.load %arg3[%arg18 + %arg13 * 5] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg19, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg9[%arg11 * 12 + %arg13 + %arg12 * 4 + %arg10 * 144 + 48] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg9[%arg11 * 12 + %arg13 + %arg12 * 4 + %arg10 * 144 + 48] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg10 = 0 to 2 {
      affine.for %arg11 = 0 to 3 {
        affine.for %arg12 = 0 to 4 {
          affine.for %arg13 = 0 to 4 {
            %0 = affine.for %arg14 = 0 to 5 iter_args(%arg15 = %cst) -> (f64) {
              %3 = affine.load %arg2[%arg14 + %arg11 * 5] : memref<?xf64>
              %4 = affine.for %arg16 = 0 to 5 iter_args(%arg17 = %arg15) -> (f64) {
                %5 = affine.load %arg3[%arg16 + %arg12 * 5] : memref<?xf64>
                %6 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %arg17) -> (f64) {
                  %7 = affine.load %alloca_34[%arg10, 2, %arg14, %arg16, %arg18] : memref<2x3x5x5x5xf64>
                  %8 = affine.load %arg3[%arg18 + %arg13 * 5] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg19, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg9[%arg11 * 16 + %arg13 + %arg12 * 4 + %arg10 * 144 + 96] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg9[%arg11 * 16 + %arg13 + %arg12 * 4 + %arg10 * 144 + 96] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_curlcurl_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_4 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_9 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_13 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_14 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_15 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_16 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_17 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_18 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_21 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_22 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_23 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_24 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_25 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_26 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_27 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_28 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_29 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_30 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_31 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_32 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_33 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg7[%arg10 * 12 + %arg13 + %arg11 * 3 + %arg9 * 144] : memref<?xf64>
              %2 = affine.load %arg0[%arg13 + %arg12 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_33[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_33[%arg9, %arg10, %arg13, %arg12] : memref<2x4x4x5xf64>
              %2 = affine.load %arg4[%arg13 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_28[%arg9, %arg10, %arg11, %arg12] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_33[%arg9, %arg10, %arg13, %arg12] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg13 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_27[%arg9, %arg10, %arg11, %arg12] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_28[%arg9, %arg13, %arg11, %arg12] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg13 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_22[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_27[%arg9, %arg13, %arg11, %arg12] : memref<2x4x5x5xf64>
              %2 = affine.load %arg4[%arg13 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_21[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg7[%arg10 * 12 + %arg13 + %arg11 * 4 + %arg9 * 144 + 48] : memref<?xf64>
              %2 = affine.load %arg4[%arg13 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_32[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg7[%arg10 * 12 + %arg13 + %arg11 * 4 + %arg9 * 144 + 48] : memref<?xf64>
              %2 = affine.load %arg1[%arg13 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_31[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_32[%arg9, %arg10, %arg13, %arg12] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg13 + %arg11 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_26[%arg9, %arg10, %arg11, %arg12] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_31[%arg9, %arg10, %arg13, %arg12] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg13 + %arg11 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_25[%arg9, %arg10, %arg11, %arg12] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_26[%arg9, %arg13, %arg11, %arg12] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg13 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg9, %arg13, %arg11, %arg12] : memref<2x4x5x5xf64>
              %2 = affine.load %arg4[%arg13 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg7[%arg10 * 16 + %arg13 + %arg11 * 4 + %arg9 * 144 + 96] : memref<?xf64>
              %2 = affine.load %arg4[%arg13 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_30[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg7[%arg10 * 16 + %arg13 + %arg11 * 4 + %arg9 * 144 + 96] : memref<?xf64>
              %2 = affine.load %arg1[%arg13 + %arg12 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_29[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_30[%arg9, %arg10, %arg13, %arg12] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg13 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_24[%arg9, %arg10, %arg11, %arg12] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_29[%arg9, %arg10, %arg13, %arg12] : memref<2x4x4x5xf64>
              %2 = affine.load %arg4[%arg13 + %arg11 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_23[%arg9, %arg10, %arg11, %arg12] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_24[%arg9, %arg13, %arg11, %arg12] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg13 + %arg10 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_18[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_23[%arg9, %arg13, %arg11, %arg12] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg13 + %arg10 * 3] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_17[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 125] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 375] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 500] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg2[%arg13 + %arg12 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg14, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_16[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_16[%arg9, %arg10, %arg13, %arg12] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg13 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg9, %arg10, %arg11, %arg12] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg9, %arg13, %arg11, %arg12] : memref<2x5x4x4xf64>
              %2 = affine.load %arg5[%arg13 + %arg10 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 250] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 500] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 625] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg2[%arg13 + %arg12 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg14, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_15[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg9, %arg10, %arg13, %arg12] : memref<2x5x5x4xf64>
              %2 = affine.load %arg5[%arg13 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg9, %arg10, %arg11, %arg12] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_9[%arg9, %arg13, %arg11, %arg12] : memref<2x5x4x4xf64>
              %2 = affine.load %arg3[%arg13 + %arg10 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 250] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 500] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 625] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg5[%arg13 + %arg12 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg14, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_14[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_14[%arg9, %arg10, %arg13, %arg12] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg13 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg9, %arg10, %arg11, %arg12] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg9, %arg13, %arg11, %arg12] : memref<2x5x4x4xf64>
              %2 = affine.load %arg3[%arg13 + %arg10 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 125] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 250] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg3[%arg13 + %arg12 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg14, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_13[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_13[%arg9, %arg10, %arg13, %arg12] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg13 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_7[%arg9, %arg10, %arg11, %arg12] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_7[%arg9, %arg13, %arg11, %arg12] : memref<2x5x4x4xf64>
              %2 = affine.load %arg5[%arg13 + %arg10 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 125] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 250] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg3[%arg13 + %arg12 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg14, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_12[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg9, %arg10, %arg13, %arg12] : memref<2x5x5x4xf64>
              %2 = affine.load %arg5[%arg13 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg9, %arg10, %arg11, %arg12] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_6[%arg9, %arg13, %arg11, %arg12] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg13 + %arg10 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 125] : memref<?xf64>
              %2 = affine.load %alloca_17[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %3 = affine.load %alloca_19[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %4 = arith.subf %2, %3 : f64
              %5 = arith.mulf %1, %4 : f64
              %6 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 375] : memref<?xf64>
              %7 = affine.load %alloca_21[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %8 = affine.load %alloca_18[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %9 = arith.subf %7, %8 : f64
              %10 = arith.mulf %6, %9 : f64
              %11 = arith.addf %5, %10 : f64
              %12 = affine.load %arg6[%arg9 * 750 + %arg13 + %arg10 * 25 + %arg11 * 5 + 500] : memref<?xf64>
              %13 = affine.load %alloca_20[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %14 = affine.load %alloca_22[%arg9, %arg10, %arg11, %arg13] : memref<2x5x5x5xf64>
              %15 = arith.subf %13, %14 : f64
              %16 = arith.mulf %12, %15 : f64
              %17 = arith.addf %11, %16 : f64
              %18 = affine.load %arg5[%arg13 + %arg12 * 5] : memref<?xf64>
              %19 = arith.mulf %17, %18 : f64
              %20 = arith.addf %arg14, %19 : f64
              affine.yield %20 : f64
            }
            affine.store %0, %alloca_11[%arg9, %arg10, %arg11, %arg12] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg9, %arg10, %arg13, %arg12] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg13 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg9, %arg10, %arg11, %arg12] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg9, %arg13, %arg11, %arg12] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg13 + %arg10 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg14, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            %0 = affine.load %alloca_4[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_3[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
            %2 = arith.subf %0, %1 : f64
            %3 = affine.load %arg8[%arg10 * 12 + %arg12 + %arg11 * 3 + %arg9 * 144] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg8[%arg10 * 12 + %arg12 + %arg11 * 3 + %arg9 * 144] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.load %alloca_2[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_1[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
            %2 = arith.subf %0, %1 : f64
            %3 = affine.load %arg8[%arg10 * 12 + %arg12 + %arg11 * 4 + %arg9 * 144 + 48] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg8[%arg10 * 12 + %arg12 + %arg11 * 4 + %arg9 * 144 + 48] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            %0 = affine.load %alloca_0[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca[%arg9, %arg10, %arg11, %arg12] : memref<2x4x4x4xf64>
            %2 = arith.subf %0, %1 : f64
            %3 = affine.load %arg8[%arg10 * 16 + %arg12 + %arg11 * 4 + %arg9 * 144 + 96] : memref<?xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg8[%arg10 * 16 + %arg12 + %arg11 * 4 + %arg9 * 144 + 96] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_hcurl_mass_apply_3d_direct(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x3x5x5x5xf64>
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg1[%arg11 + %arg8 * 4] : memref<?xf64>
              %2 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %arg12) -> (f64) {
                %3 = affine.load %arg1[%arg13 + %arg9 * 4] : memref<?xf64>
                %4 = affine.for %arg15 = 0 to 3 iter_args(%arg16 = %arg14) -> (f64) {
                  %5 = affine.load %arg5[%arg11 * 12 + %arg15 + %arg13 * 3 + %arg7 * 144] : memref<?xf64>
                  %6 = affine.load %arg0[%arg15 + %arg10 * 3] : memref<?xf64>
                  %7 = arith.mulf %5, %6 : f64
                  %8 = arith.mulf %7, %3 : f64
                  %9 = arith.mulf %8, %1 : f64
                  %10 = arith.addf %arg16, %9 : f64
                  affine.yield %10 : f64
                }
                affine.yield %4 : f64
              }
              affine.yield %2 : f64
            }
            affine.store %0, %alloca[%arg7, 0, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg1[%arg11 + %arg8 * 4] : memref<?xf64>
              %2 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %arg12) -> (f64) {
                %3 = affine.load %arg0[%arg13 + %arg9 * 3] : memref<?xf64>
                %4 = affine.for %arg15 = 0 to 4 iter_args(%arg16 = %arg14) -> (f64) {
                  %5 = affine.load %arg5[%arg11 * 12 + %arg15 + %arg13 * 4 + %arg7 * 144 + 48] : memref<?xf64>
                  %6 = affine.load %arg1[%arg15 + %arg10 * 4] : memref<?xf64>
                  %7 = arith.mulf %5, %6 : f64
                  %8 = arith.mulf %7, %3 : f64
                  %9 = arith.mulf %8, %1 : f64
                  %10 = arith.addf %arg16, %9 : f64
                  affine.yield %10 : f64
                }
                affine.yield %4 : f64
              }
              affine.yield %2 : f64
            }
            affine.store %0, %alloca[%arg7, 1, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg11 + %arg8 * 3] : memref<?xf64>
              %2 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %arg12) -> (f64) {
                %3 = affine.load %arg1[%arg13 + %arg9 * 4] : memref<?xf64>
                %4 = affine.for %arg15 = 0 to 4 iter_args(%arg16 = %arg14) -> (f64) {
                  %5 = affine.load %arg5[%arg11 * 16 + %arg15 + %arg13 * 4 + %arg7 * 144 + 96] : memref<?xf64>
                  %6 = affine.load %arg1[%arg15 + %arg10 * 4] : memref<?xf64>
                  %7 = arith.mulf %5, %6 : f64
                  %8 = arith.mulf %7, %3 : f64
                  %9 = arith.mulf %8, %1 : f64
                  %10 = arith.addf %arg16, %9 : f64
                  affine.yield %10 : f64
                }
                affine.yield %4 : f64
              }
              affine.yield %2 : f64
            }
            affine.store %0, %alloca[%arg7, 2, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.load %alloca[%arg7, 0, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
            %1 = affine.load %alloca[%arg7, 1, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
            %2 = affine.load %alloca[%arg7, 2, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
            %3 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750] : memref<?xf64>
            %4 = arith.mulf %3, %0 : f64
            %5 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 125] : memref<?xf64>
            %6 = arith.mulf %5, %1 : f64
            %7 = arith.addf %4, %6 : f64
            %8 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 250] : memref<?xf64>
            %9 = arith.mulf %8, %2 : f64
            %10 = arith.addf %7, %9 : f64
            affine.store %10, %alloca[%arg7, 0, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
            %11 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 125] : memref<?xf64>
            %12 = arith.mulf %11, %0 : f64
            %13 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 375] : memref<?xf64>
            %14 = arith.mulf %13, %1 : f64
            %15 = arith.addf %12, %14 : f64
            %16 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 500] : memref<?xf64>
            %17 = arith.mulf %16, %2 : f64
            %18 = arith.addf %15, %17 : f64
            affine.store %18, %alloca[%arg7, 1, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
            %19 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 250] : memref<?xf64>
            %20 = arith.mulf %19, %0 : f64
            %21 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 500] : memref<?xf64>
            %22 = arith.mulf %21, %1 : f64
            %23 = arith.addf %20, %22 : f64
            %24 = affine.load %arg4[%arg8 * 25 + %arg10 + %arg9 * 5 + %arg7 * 750 + 625] : memref<?xf64>
            %25 = arith.mulf %24, %2 : f64
            %26 = arith.addf %23, %25 : f64
            affine.store %26, %alloca[%arg7, 2, %arg8, %arg9, %arg10] : memref<2x3x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 3 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %arg3[%arg11 + %arg8 * 5] : memref<?xf64>
              %4 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %arg12) -> (f64) {
                %5 = affine.load %arg3[%arg13 + %arg9 * 5] : memref<?xf64>
                %6 = affine.for %arg15 = 0 to 5 iter_args(%arg16 = %arg14) -> (f64) {
                  %7 = affine.load %alloca[%arg7, 0, %arg11, %arg13, %arg15] : memref<2x3x5x5x5xf64>
                  %8 = affine.load %arg2[%arg15 + %arg10 * 5] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg16, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg6[%arg8 * 12 + %arg10 + %arg9 * 3 + %arg7 * 144] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 12 + %arg10 + %arg9 * 3 + %arg7 * 144] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 3 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %arg3[%arg11 + %arg8 * 5] : memref<?xf64>
              %4 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %arg12) -> (f64) {
                %5 = affine.load %arg2[%arg13 + %arg9 * 5] : memref<?xf64>
                %6 = affine.for %arg15 = 0 to 5 iter_args(%arg16 = %arg14) -> (f64) {
                  %7 = affine.load %alloca[%arg7, 1, %arg11, %arg13, %arg15] : memref<2x3x5x5x5xf64>
                  %8 = affine.load %arg3[%arg15 + %arg10 * 5] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg16, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg6[%arg8 * 12 + %arg10 + %arg9 * 4 + %arg7 * 144 + 48] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 12 + %arg10 + %arg9 * 4 + %arg7 * 144 + 48] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %3 = affine.load %arg2[%arg11 + %arg8 * 5] : memref<?xf64>
              %4 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %arg12) -> (f64) {
                %5 = affine.load %arg3[%arg13 + %arg9 * 5] : memref<?xf64>
                %6 = affine.for %arg15 = 0 to 5 iter_args(%arg16 = %arg14) -> (f64) {
                  %7 = affine.load %alloca[%arg7, 2, %arg11, %arg13, %arg15] : memref<2x3x5x5x5xf64>
                  %8 = affine.load %arg3[%arg15 + %arg10 * 5] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg16, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg6[%arg8 * 16 + %arg10 + %arg9 * 4 + %arg7 * 144 + 96] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg6[%arg8 * 16 + %arg10 + %arg9 * 4 + %arg7 * 144 + 96] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
