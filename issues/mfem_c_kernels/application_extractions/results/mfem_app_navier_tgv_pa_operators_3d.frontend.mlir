module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_navier_tgv_pa_operators_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>, %arg12: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<20xf64>
    affine.for %arg13 = 0 to 4 {
      affine.for %arg14 = 0 to 5 {
        %0 = affine.load %arg0[%arg13 + %arg14 * 4] : memref<?xf64>
        affine.store %0, %alloca[%arg14 + %arg13 * 5] : memref<20xf64>
      }
    }
    %alloca_0 = memref.alloca() : memref<128xf64>
    %alloca_1 = memref.alloca() : memref<128xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %0, %alloca_1[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %1 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %1, %alloca_0[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_2 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_5 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %2 = affine.load %alloca_1[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %alloca_6[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %2 = affine.load %alloca_5[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.load %arg5[%arg13 * 125 + %arg16 + %arg14 * 25 + %arg15 * 5] : memref<?xf64>
            %1 = affine.load %alloca_4[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_4[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg17 + %arg16 * 5] : memref<20xf64>
              %2 = affine.load %alloca_4[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg17 + %arg15 * 5] : memref<20xf64>
              %2 = affine.load %alloca_3[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg17 + %arg14 * 5] : memref<20xf64>
              %4 = affine.load %alloca_2[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg18, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %alloca_0[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_0[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_0[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_7 = memref.alloca() : memref<128xf64>
    %alloca_8 = memref.alloca() : memref<128xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_8[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %1 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %1, %alloca_7[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_9 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_13 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %2 = affine.load %alloca_8[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %alloca_13[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_12[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %2 = affine.load %alloca_12[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.load %arg5[%arg13 * 125 + %arg16 + %arg14 * 25 + %arg15 * 5] : memref<?xf64>
            %1 = affine.load %alloca_11[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_11[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg17 + %arg16 * 5] : memref<20xf64>
              %2 = affine.load %alloca_11[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg17 + %arg15 * 5] : memref<20xf64>
              %2 = affine.load %alloca_10[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg17 + %arg14 * 5] : memref<20xf64>
              %4 = affine.load %alloca_9[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg18, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %alloca_7[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_7[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_7[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_14 = memref.alloca() : memref<128xf64>
    %alloca_15 = memref.alloca() : memref<128xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_15[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %1 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %1, %alloca_14[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_16 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_17 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_18 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %2 = affine.load %alloca_15[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %alloca_20[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %2 = affine.load %alloca_19[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_18[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.load %arg5[%arg13 * 125 + %arg16 + %arg14 * 25 + %arg15 * 5] : memref<?xf64>
            %1 = affine.load %alloca_18[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_18[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg17 + %arg16 * 5] : memref<20xf64>
              %2 = affine.load %alloca_18[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_17[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg17 + %arg15 * 5] : memref<20xf64>
              %2 = affine.load %alloca_17[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_16[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg17 + %arg14 * 5] : memref<20xf64>
              %4 = affine.load %alloca_16[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg18, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %alloca_14[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_14[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_14[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    %alloca_21 = memref.alloca() : memref<20xf64>
    %alloca_22 = memref.alloca() : memref<20xf64>
    affine.for %arg13 = 0 to 4 {
      affine.for %arg14 = 0 to 5 {
        %0 = affine.load %arg0[%arg13 + %arg14 * 4] : memref<?xf64>
        affine.store %0, %alloca_22[%arg14 + %arg13 * 5] : memref<20xf64>
        %1 = affine.load %arg1[%arg13 + %arg14 * 4] : memref<?xf64>
        affine.store %1, %alloca_21[%arg14 + %arg13 * 5] : memref<20xf64>
      }
    }
    %alloca_23 = memref.alloca() : memref<128xf64>
    %alloca_24 = memref.alloca() : memref<128xf64>
    %alloca_25 = memref.alloca() : memref<1500xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 6 {
        affine.for %arg15 = 0 to 125 {
          %0 = affine.load %arg6[%arg15 + %arg13 * 2250 + %arg14 * 125] : memref<?xf64>
          affine.store %0, %alloca_25[%arg15 + %arg13 * 750 + %arg14 * 125] : memref<1500xf64>
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %0, %alloca_24[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %1 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %1, %alloca_23[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_26 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_27 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_28 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_29 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_30 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_31 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_32 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_33 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_34 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_35 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_36 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_37 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_38 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_39 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_40 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_41 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_42 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_24[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_42[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_24[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_41[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_41[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_40[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_42[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_39[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_42[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_38[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_40[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_37[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_39[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_36[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_38[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_35[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 750] : memref<1500xf64>
              %2 = affine.load %alloca_37[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<1500xf64>
              %5 = affine.load %alloca_36[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<1500xf64>
              %9 = affine.load %alloca_35[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_21[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_34[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<1500xf64>
              %2 = affine.load %alloca_37[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 375] : memref<1500xf64>
              %5 = affine.load %alloca_36[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<1500xf64>
              %9 = affine.load %alloca_35[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_22[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_33[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<1500xf64>
              %2 = affine.load %alloca_37[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<1500xf64>
              %5 = affine.load %alloca_36[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_25[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 625] : memref<1500xf64>
              %9 = affine.load %alloca_35[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_22[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_32[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_34[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_31[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_33[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_21[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_30[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_32[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_29[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_31[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_28[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_30[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_27[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_29[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_21[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_26[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_28[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_27[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_26[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %alloca_23[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %alloca_23[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_23[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_43 = memref.alloca() : memref<128xf64>
    %alloca_44 = memref.alloca() : memref<128xf64>
    %alloca_45 = memref.alloca() : memref<1500xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 6 {
        affine.for %arg15 = 0 to 125 {
          %0 = affine.load %arg6[%arg15 + %arg13 * 2250 + %arg14 * 125 + 750] : memref<?xf64>
          affine.store %0, %alloca_45[%arg15 + %arg13 * 750 + %arg14 * 125] : memref<1500xf64>
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_44[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %1 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %1, %alloca_43[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_46 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_47 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_48 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_49 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_50 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_51 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_52 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_53 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_54 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_55 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_56 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_57 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_58 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_59 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_60 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_61 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_62 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_44[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_62[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_44[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_61[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_61[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_60[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_62[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_59[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_62[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_58[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_60[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_57[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_59[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_56[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_58[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_55[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 750] : memref<1500xf64>
              %2 = affine.load %alloca_57[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<1500xf64>
              %5 = affine.load %alloca_56[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<1500xf64>
              %9 = affine.load %alloca_55[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_21[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_54[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<1500xf64>
              %2 = affine.load %alloca_57[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 375] : memref<1500xf64>
              %5 = affine.load %alloca_56[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<1500xf64>
              %9 = affine.load %alloca_55[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_22[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_53[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<1500xf64>
              %2 = affine.load %alloca_57[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<1500xf64>
              %5 = affine.load %alloca_56[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_45[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 625] : memref<1500xf64>
              %9 = affine.load %alloca_55[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_22[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_52[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_54[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_51[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_53[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_21[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_50[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_52[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_49[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_51[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_48[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_50[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_47[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_49[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_21[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_46[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_48[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_47[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_46[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %alloca_43[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %alloca_43[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_43[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_63 = memref.alloca() : memref<128xf64>
    %alloca_64 = memref.alloca() : memref<128xf64>
    %alloca_65 = memref.alloca() : memref<1500xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 6 {
        affine.for %arg15 = 0 to 125 {
          %0 = affine.load %arg6[%arg15 + %arg13 * 2250 + %arg14 * 125 + 1500] : memref<?xf64>
          affine.store %0, %alloca_65[%arg15 + %arg13 * 750 + %arg14 * 125] : memref<1500xf64>
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_64[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %1 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %1, %alloca_63[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_66 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_67 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_68 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_69 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_70 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_71 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_72 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_73 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_74 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_75 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_76 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_77 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_78 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_79 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_80 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_81 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_82 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_64[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_82[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_64[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_81[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_81[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_80[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_82[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_79[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_82[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_78[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_80[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_77[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_79[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_76[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_78[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_75[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 750] : memref<1500xf64>
              %2 = affine.load %alloca_77[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<1500xf64>
              %5 = affine.load %alloca_76[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<1500xf64>
              %9 = affine.load %alloca_75[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_21[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_74[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<1500xf64>
              %2 = affine.load %alloca_77[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 375] : memref<1500xf64>
              %5 = affine.load %alloca_76[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<1500xf64>
              %9 = affine.load %alloca_75[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_22[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_73[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<1500xf64>
              %2 = affine.load %alloca_77[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<1500xf64>
              %5 = affine.load %alloca_76[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_65[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 625] : memref<1500xf64>
              %9 = affine.load %alloca_75[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_22[%arg17 + %arg16 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_72[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_74[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_71[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_73[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_21[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_70[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_72[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_69[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_71[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_68[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_70[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_22[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_67[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_69[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_21[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_66[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_68[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_67[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_66[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %alloca_63[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %alloca_63[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_63[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    %alloca_83 = memref.alloca() : memref<750xf64>
    %alloca_84 = memref.alloca() : memref<2250xf64>
    %alloca_85 = memref.alloca() : memref<750xf64>
    %alloca_86 = memref.alloca() : memref<20xf64>
    affine.for %arg13 = 0 to 4 {
      affine.for %arg14 = 0 to 5 {
        %0 = affine.load %arg0[%arg13 + %arg14 * 4] : memref<?xf64>
        affine.store %0, %alloca_86[%arg14 + %arg13 * 5] : memref<20xf64>
      }
    }
    %alloca_87 = memref.alloca() : memref<750xf64>
    %alloca_88 = memref.alloca() : memref<250xf64>
    %alloca_89 = memref.alloca() : memref<128xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %0, %alloca_89[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_90 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_91 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_89[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_91[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_91[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_90[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_90[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_88[%arg14 * 25 + %arg16 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
          }
        }
      }
    }
    %alloca_92 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_93 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_94 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_95 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_96 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_89[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_96[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_89[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_95[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_95[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_94[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_96[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_93[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_96[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_92[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_94[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_87[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_93[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_87[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_92[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_87[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.load %alloca_88[%arg14 * 25 + %arg16 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
            affine.store %0, %alloca_85[%arg14 * 25 + %arg16 + %arg13 * 375 + %arg15 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 3 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            affine.for %arg17 = 0 to 5 {
              %0 = affine.load %alloca_87[%arg13 * 375 + %arg14 * 125 + %arg15 + %arg17 * 25 + %arg16 * 5] : memref<750xf64>
              affine.store %0, %alloca_84[%arg13 * 1125 + %arg15 * 25 + %arg17 + %arg16 * 5 + %arg14 * 125] : memref<2250xf64>
            }
          }
        }
      }
    }
    %alloca_97 = memref.alloca() : memref<750xf64>
    %alloca_98 = memref.alloca() : memref<250xf64>
    %alloca_99 = memref.alloca() : memref<128xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_99[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_100 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_101 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_99[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_101[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_101[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_100[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_100[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_98[%arg14 * 25 + %arg16 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
          }
        }
      }
    }
    %alloca_102 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_103 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_104 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_105 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_106 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_99[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_106[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_99[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_105[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_105[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_104[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_106[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_103[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_106[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_102[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_104[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_97[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_103[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_97[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_102[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_97[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.load %alloca_98[%arg14 * 25 + %arg16 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
            affine.store %0, %alloca_85[%arg14 * 25 + %arg16 + %arg13 * 375 + %arg15 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 3 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            affine.for %arg17 = 0 to 5 {
              %0 = affine.load %alloca_97[%arg13 * 375 + %arg14 * 125 + %arg15 + %arg17 * 25 + %arg16 * 5] : memref<750xf64>
              affine.store %0, %alloca_84[%arg13 * 1125 + %arg15 * 25 + %arg17 + %arg16 * 5 + %arg14 * 125 + 375] : memref<2250xf64>
            }
          }
        }
      }
    }
    %alloca_107 = memref.alloca() : memref<750xf64>
    %alloca_108 = memref.alloca() : memref<250xf64>
    %alloca_109 = memref.alloca() : memref<128xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg9[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_109[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_110 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_111 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_109[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_111[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_111[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_110[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_110[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_108[%arg14 * 25 + %arg16 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
          }
        }
      }
    }
    %alloca_112 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_113 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_114 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_115 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_116 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_109[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_116[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_109[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_115[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_115[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_114[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_116[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_113[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_116[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_112[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_114[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_107[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_113[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_107[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_112[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_107[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.load %alloca_108[%arg14 * 25 + %arg16 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
            affine.store %0, %alloca_85[%arg14 * 25 + %arg16 + %arg13 * 375 + %arg15 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 3 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            affine.for %arg17 = 0 to 5 {
              %0 = affine.load %alloca_107[%arg13 * 375 + %arg14 * 125 + %arg15 + %arg17 * 25 + %arg16 * 5] : memref<750xf64>
              affine.store %0, %alloca_84[%arg13 * 1125 + %arg15 * 25 + %arg17 + %arg16 * 5 + %arg14 * 125 + 750] : memref<2250xf64>
            }
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 3 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            affine.for %arg17 = 0 to 5 {
              %0 = affine.for %arg18 = 0 to 3 iter_args(%arg19 = %cst) -> (f64) {
                %1 = affine.load %alloca_85[%arg15 * 25 + %arg13 * 375 + %arg17 + %arg16 * 5 + %arg18 * 125] : memref<750xf64>
                %2 = affine.for %arg20 = 0 to 3 iter_args(%arg21 = %arg19) -> (f64) {
                  %3 = affine.load %alloca_84[%arg13 * 1125 + %arg14 * 375 + %arg15 * 25 + %arg17 + %arg16 * 5 + %arg20 * 125] : memref<2250xf64>
                  %4 = arith.mulf %1, %3 : f64
                  %5 = affine.load %arg7[%arg13 * 1125 + %arg18 * 375 + %arg15 * 25 + %arg17 + %arg16 * 5 + %arg20 * 125] : memref<?xf64>
                  %6 = arith.mulf %4, %5 : f64
                  %7 = arith.addf %arg21, %6 : f64
                  affine.yield %7 : f64
                }
                affine.yield %2 : f64
              }
              affine.store %0, %alloca_83[%arg15 * 25 + %arg13 * 375 + %arg17 + %arg16 * 5 + %arg14 * 125] : memref<750xf64>
            }
          }
        }
      }
    }
    %alloca_117 = memref.alloca() : memref<128xf64>
    %alloca_118 = memref.alloca() : memref<250xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 125 {
        %0 = affine.load %alloca_83[%arg14 + %arg13 * 375] : memref<750xf64>
        affine.store %0, %alloca_118[%arg14 + %arg13 * 125] : memref<250xf64>
      }
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %0, %alloca_117[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_119 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_120 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_121 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_118[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg16 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_121[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_121[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_120[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_120[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_119[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_119[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_117[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_117[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_117[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_122 = memref.alloca() : memref<128xf64>
    %alloca_123 = memref.alloca() : memref<250xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 125 {
        %0 = affine.load %alloca_83[%arg14 + %arg13 * 375 + 125] : memref<750xf64>
        affine.store %0, %alloca_123[%arg14 + %arg13 * 125] : memref<250xf64>
      }
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_122[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_124 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_125 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_126 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_123[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg16 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_126[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_126[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_125[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_125[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_124[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_124[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_122[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_122[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_122[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_127 = memref.alloca() : memref<128xf64>
    %alloca_128 = memref.alloca() : memref<250xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 125 {
        %0 = affine.load %alloca_83[%arg14 + %arg13 * 375 + 250] : memref<750xf64>
        affine.store %0, %alloca_128[%arg14 + %arg13 * 125] : memref<250xf64>
      }
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_127[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_129 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_130 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_131 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_128[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg16 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_131[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_131[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_130[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_130[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_86[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_129[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_129[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_127[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_127[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_127[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    %alloca_132 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_133 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_134 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_135 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_136 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_137 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_138 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_139 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_140 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_141 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_142 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_143 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_144 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_145 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_146 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_147 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_148 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg10[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_148[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg10[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_147[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_147[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_146[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_148[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_145[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_148[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_144[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_146[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_143[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_145[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_142[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_144[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_141[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 750] : memref<?xf64>
              %2 = affine.load %alloca_143[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<?xf64>
              %5 = affine.load %alloca_142[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<?xf64>
              %9 = affine.load %alloca_141[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %arg3[%arg17 + %arg16 * 5] : memref<?xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_140[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 125] : memref<?xf64>
              %2 = affine.load %alloca_143[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 375] : memref<?xf64>
              %5 = affine.load %alloca_142[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<?xf64>
              %9 = affine.load %alloca_141[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %arg2[%arg17 + %arg16 * 5] : memref<?xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_139[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 250] : memref<?xf64>
              %2 = affine.load %alloca_143[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 500] : memref<?xf64>
              %5 = affine.load %alloca_142[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg4[%arg14 * 25 + %arg17 + %arg13 * 750 + %arg15 * 5 + 625] : memref<?xf64>
              %9 = affine.load %alloca_141[%arg13, %arg14, %arg15, %arg17] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %arg2[%arg17 + %arg16 * 5] : memref<?xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg18, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_138[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_140[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg17 + %arg15 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_137[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_139[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg17 + %arg15 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_136[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_138[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg17 + %arg15 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_135[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_137[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg17 + %arg14 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_134[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_136[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg17 + %arg14 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_133[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_135[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %arg3[%arg17 + %arg14 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_132[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_134[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_133[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_132[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %arg12[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %arg12[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_149 = memref.alloca() : memref<750xf64>
    %alloca_150 = memref.alloca() : memref<750xf64>
    %alloca_151 = memref.alloca() : memref<20xf64>
    affine.for %arg13 = 0 to 4 {
      affine.for %arg14 = 0 to 5 {
        %0 = affine.load %arg0[%arg13 + %arg14 * 4] : memref<?xf64>
        affine.store %0, %alloca_151[%arg14 + %arg13 * 5] : memref<20xf64>
      }
    }
    %alloca_152 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_153 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_154 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_155 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_156 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg10[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %arg0[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_156[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %arg10[%arg13 * 64 + %arg17 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
              %2 = affine.load %arg1[%arg17 + %arg16 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_155[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_155[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_154[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_156[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_153[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_156[%arg13, %arg14, %arg17, %arg16] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg15 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_152[%arg13, %arg14, %arg15, %arg16] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_154[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_150[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_153[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_150[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 4 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_152[%arg13, %arg17, %arg15, %arg16] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg17 + %arg14 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_150[%arg13 * 375 + %arg14 + %arg16 * 25 + %arg15 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 3 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            affine.for %arg17 = 0 to 5 {
              %0 = affine.for %arg18 = 0 to 3 iter_args(%arg19 = %cst) -> (f64) {
                %1 = affine.load %alloca_150[%arg13 * 375 + %arg18 * 125 + %arg15 + %arg17 * 25 + %arg16 * 5] : memref<750xf64>
                %2 = affine.load %arg8[%arg13 * 1125 + %arg14 * 375 + %arg15 * 25 + %arg17 + %arg16 * 5 + %arg18 * 125] : memref<?xf64>
                %3 = arith.mulf %1, %2 : f64
                %4 = arith.addf %arg19, %3 : f64
                affine.yield %4 : f64
              }
              affine.store %0, %alloca_149[%arg15 * 25 + %arg13 * 375 + %arg17 + %arg16 * 5 + %arg14 * 125] : memref<750xf64>
            }
          }
        }
      }
    }
    %alloca_157 = memref.alloca() : memref<128xf64>
    %alloca_158 = memref.alloca() : memref<250xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 125 {
        %0 = affine.load %alloca_149[%arg14 + %arg13 * 375] : memref<750xf64>
        affine.store %0, %alloca_158[%arg14 + %arg13 * 125] : memref<250xf64>
      }
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            affine.store %0, %alloca_157[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_159 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_160 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_161 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_158[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg16 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_161[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_161[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_160[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_160[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_159[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_159[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_157[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_157[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_157[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_162 = memref.alloca() : memref<128xf64>
    %alloca_163 = memref.alloca() : memref<250xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 125 {
        %0 = affine.load %alloca_149[%arg14 + %arg13 * 375 + 125] : memref<750xf64>
        affine.store %0, %alloca_163[%arg14 + %arg13 * 125] : memref<250xf64>
      }
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_162[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_164 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_165 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_166 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_163[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg16 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_166[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_166[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_165[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_165[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_164[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_164[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_162[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_162[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_162[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_167 = memref.alloca() : memref<128xf64>
    %alloca_168 = memref.alloca() : memref<250xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 125 {
        %0 = affine.load %alloca_149[%arg14 + %arg13 * 375 + 250] : memref<750xf64>
        affine.store %0, %alloca_168[%arg14 + %arg13 * 125] : memref<250xf64>
      }
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_167[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_169 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_170 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_171 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_168[%arg14 * 25 + %arg17 + %arg15 * 5 + %arg13 * 125] : memref<250xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg16 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_171[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_171[%arg13, %arg14, %arg17, %arg16] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg15 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_170[%arg13, %arg14, %arg15, %arg16] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.load %alloca_170[%arg13, %arg17, %arg15, %arg16] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_151[%arg17 + %arg14 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg18, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_169[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_169[%arg13, %arg14, %arg15, %arg16] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_167[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_167[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.load %alloca_167[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<128xf64>
            affine.store %0, %arg11[%arg13 * 192 + %arg16 + %arg14 * 16 + %arg15 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    %alloca_172 = memref.alloca() : memref<2x5x5x5xf64>
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 5 {
        affine.for %arg15 = 0 to 5 {
          affine.for %arg16 = 0 to 5 {
            %0 = affine.for %arg17 = 0 to 3 iter_args(%arg18 = %cst) -> (f64) {
              %1 = affine.for %arg19 = 0 to 4 iter_args(%arg20 = %arg18) -> (f64) {
                %2 = affine.load %arg0[%arg19 + %arg14 * 4] : memref<?xf64>
                %3 = affine.load %arg8[%arg14 * 25 + %arg13 * 1125 + %arg16 + %arg15 * 5 + %arg17 * 375] : memref<?xf64>
                %4 = affine.load %arg8[%arg13 * 1125 + %arg17 * 375 + %arg16 + %arg14 * 25 + %arg15 * 5 + 125] : memref<?xf64>
                %5 = affine.load %arg1[%arg19 + %arg14 * 4] : memref<?xf64>
                %6 = affine.load %arg8[%arg13 * 1125 + %arg17 * 375 + %arg16 + %arg14 * 25 + %arg15 * 5 + 250] : memref<?xf64>
                %7 = affine.for %arg21 = 0 to 4 iter_args(%arg22 = %arg20) -> (f64) {
                  %8 = affine.load %arg0[%arg21 + %arg15 * 4] : memref<?xf64>
                  %9 = affine.load %arg1[%arg21 + %arg15 * 4] : memref<?xf64>
                  %10 = affine.for %arg23 = 0 to 4 iter_args(%arg24 = %arg22) -> (f64) {
                    %11 = affine.load %arg9[%arg13 * 192 + %arg17 * 64 + %arg23 + %arg19 * 16 + %arg21 * 4] : memref<?xf64>
                    %12 = affine.load %arg1[%arg23 + %arg16 * 4] : memref<?xf64>
                    %13 = arith.mulf %11, %12 : f64
                    %14 = arith.mulf %13, %8 : f64
                    %15 = arith.mulf %14, %2 : f64
                    %16 = arith.mulf %15, %3 : f64
                    %17 = arith.addf %arg24, %16 : f64
                    %18 = affine.load %arg0[%arg23 + %arg16 * 4] : memref<?xf64>
                    %19 = arith.mulf %11, %18 : f64
                    %20 = arith.mulf %19, %9 : f64
                    %21 = arith.mulf %20, %2 : f64
                    %22 = arith.mulf %21, %4 : f64
                    %23 = arith.addf %17, %22 : f64
                    %24 = arith.mulf %19, %8 : f64
                    %25 = arith.mulf %24, %5 : f64
                    %26 = arith.mulf %25, %6 : f64
                    %27 = arith.addf %23, %26 : f64
                    affine.yield %27 : f64
                  }
                  affine.yield %10 : f64
                }
                affine.yield %7 : f64
              }
              affine.yield %1 : f64
            }
            affine.store %0, %alloca_172[%arg13, %arg14, %arg15, %arg16] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg13 = 0 to 2 {
      affine.for %arg14 = 0 to 4 {
        affine.for %arg15 = 0 to 4 {
          affine.for %arg16 = 0 to 4 {
            %0 = affine.for %arg17 = 0 to 5 iter_args(%arg18 = %cst) -> (f64) {
              %3 = affine.load %arg0[%arg14 + %arg17 * 4] : memref<?xf64>
              %4 = affine.for %arg19 = 0 to 5 iter_args(%arg20 = %arg18) -> (f64) {
                %5 = affine.load %arg0[%arg15 + %arg19 * 4] : memref<?xf64>
                %6 = affine.for %arg21 = 0 to 5 iter_args(%arg22 = %arg20) -> (f64) {
                  %7 = affine.load %alloca_172[%arg13, %arg17, %arg19, %arg21] : memref<2x5x5x5xf64>
                  %8 = affine.load %arg0[%arg16 + %arg21 * 4] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg22, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg12[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg12[%arg13 * 64 + %arg16 + %arg14 * 16 + %arg15 * 4] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_vector_mass_apply_3d_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<20xf64>
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 5 {
        %0 = affine.load %arg0[%arg4 + %arg5 * 4] : memref<?xf64>
        affine.store %0, %alloca[%arg5 + %arg4 * 5] : memref<20xf64>
      }
    }
    %alloca_0 = memref.alloca() : memref<128xf64>
    %alloca_1 = memref.alloca() : memref<128xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %arg2[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
            affine.store %0, %alloca_1[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            %1 = affine.load %arg3[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
            affine.store %1, %alloca_0[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_2 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_5 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_6 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg7 * 4] : memref<?xf64>
              %2 = affine.load %alloca_1[%arg4 * 64 + %arg8 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg4, %arg5, %arg6, %arg7] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg6 * 4] : memref<?xf64>
              %2 = affine.load %alloca_6[%arg4, %arg5, %arg8, %arg7] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg4, %arg5, %arg6, %arg7] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg5 * 4] : memref<?xf64>
              %2 = affine.load %alloca_5[%arg4, %arg8, %arg6, %arg7] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.load %arg1[%arg4 * 125 + %arg7 + %arg5 * 25 + %arg6 * 5] : memref<?xf64>
            %1 = affine.load %alloca_4[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_4[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg8 + %arg7 * 5] : memref<20xf64>
              %2 = affine.load %alloca_4[%arg4, %arg5, %arg6, %arg8] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg8 + %arg6 * 5] : memref<20xf64>
              %2 = affine.load %alloca_3[%arg4, %arg5, %arg8, %arg7] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg4, %arg5, %arg6, %arg7] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg8 + %arg5 * 5] : memref<20xf64>
              %4 = affine.load %alloca_2[%arg4, %arg8, %arg6, %arg7] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg9, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %alloca_0[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_0[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %alloca_0[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            affine.store %0, %arg3[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_7 = memref.alloca() : memref<128xf64>
    %alloca_8 = memref.alloca() : memref<128xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %arg2[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_8[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            %1 = affine.load %arg3[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4 + 64] : memref<?xf64>
            affine.store %1, %alloca_7[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_9 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_13 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg7 * 4] : memref<?xf64>
              %2 = affine.load %alloca_8[%arg4 * 64 + %arg8 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg4, %arg5, %arg6, %arg7] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg6 * 4] : memref<?xf64>
              %2 = affine.load %alloca_13[%arg4, %arg5, %arg8, %arg7] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_12[%arg4, %arg5, %arg6, %arg7] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg5 * 4] : memref<?xf64>
              %2 = affine.load %alloca_12[%arg4, %arg8, %arg6, %arg7] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.load %arg1[%arg4 * 125 + %arg7 + %arg5 * 25 + %arg6 * 5] : memref<?xf64>
            %1 = affine.load %alloca_11[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_11[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg8 + %arg7 * 5] : memref<20xf64>
              %2 = affine.load %alloca_11[%arg4, %arg5, %arg6, %arg8] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg8 + %arg6 * 5] : memref<20xf64>
              %2 = affine.load %alloca_10[%arg4, %arg5, %arg8, %arg7] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg4, %arg5, %arg6, %arg7] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg8 + %arg5 * 5] : memref<20xf64>
              %4 = affine.load %alloca_9[%arg4, %arg8, %arg6, %arg7] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg9, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %alloca_7[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_7[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %alloca_7[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            affine.store %0, %arg3[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_14 = memref.alloca() : memref<128xf64>
    %alloca_15 = memref.alloca() : memref<128xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %arg2[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_15[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            %1 = affine.load %arg3[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4 + 128] : memref<?xf64>
            affine.store %1, %alloca_14[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_16 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_17 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_18 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg7 * 4] : memref<?xf64>
              %2 = affine.load %alloca_15[%arg4 * 64 + %arg8 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg4, %arg5, %arg6, %arg7] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg6 * 4] : memref<?xf64>
              %2 = affine.load %alloca_20[%arg4, %arg5, %arg8, %arg7] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg4, %arg5, %arg6, %arg7] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg8 + %arg5 * 4] : memref<?xf64>
              %2 = affine.load %alloca_19[%arg4, %arg8, %arg6, %arg7] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_18[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.load %arg1[%arg4 * 125 + %arg7 + %arg5 * 25 + %arg6 * 5] : memref<?xf64>
            %1 = affine.load %alloca_18[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_18[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg8 + %arg7 * 5] : memref<20xf64>
              %2 = affine.load %alloca_18[%arg4, %arg5, %arg6, %arg8] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_17[%arg4, %arg5, %arg6, %arg7] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg8 + %arg6 * 5] : memref<20xf64>
              %2 = affine.load %alloca_17[%arg4, %arg5, %arg8, %arg7] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_16[%arg4, %arg5, %arg6, %arg7] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %cst) -> (f64) {
              %3 = affine.load %alloca[%arg8 + %arg5 * 5] : memref<20xf64>
              %4 = affine.load %alloca_16[%arg4, %arg8, %arg6, %arg7] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg9, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %alloca_14[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_14[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %alloca_14[%arg4 * 64 + %arg7 + %arg5 * 16 + %arg6 * 4] : memref<128xf64>
            affine.store %0, %arg3[%arg4 * 192 + %arg7 + %arg5 * 16 + %arg6 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_vector_diffusion_apply_3d_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<20xf64>
    %alloca_0 = memref.alloca() : memref<20xf64>
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 5 {
        %0 = affine.load %arg0[%arg5 + %arg6 * 4] : memref<?xf64>
        affine.store %0, %alloca_0[%arg6 + %arg5 * 5] : memref<20xf64>
        %1 = affine.load %arg1[%arg5 + %arg6 * 4] : memref<?xf64>
        affine.store %1, %alloca[%arg6 + %arg5 * 5] : memref<20xf64>
      }
    }
    %alloca_1 = memref.alloca() : memref<128xf64>
    %alloca_2 = memref.alloca() : memref<128xf64>
    %alloca_3 = memref.alloca() : memref<1500xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 6 {
        affine.for %arg7 = 0 to 125 {
          %0 = affine.load %arg2[%arg7 + %arg5 * 2250 + %arg6 * 125] : memref<?xf64>
          affine.store %0, %alloca_3[%arg7 + %arg5 * 750 + %arg6 * 125] : memref<1500xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            affine.store %0, %alloca_2[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %1 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            affine.store %1, %alloca_1[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_4 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_9 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_12 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_13 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_14 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_15 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_16 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_17 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_18 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_20 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_19[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_18[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_20[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_17[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_20[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_16[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_18[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_15[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_17[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_14[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_16[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 750] : memref<1500xf64>
              %2 = affine.load %alloca_15[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 125] : memref<1500xf64>
              %5 = affine.load %alloca_14[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 250] : memref<1500xf64>
              %9 = affine.load %alloca_13[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_12[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 125] : memref<1500xf64>
              %2 = affine.load %alloca_15[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 375] : memref<1500xf64>
              %5 = affine.load %alloca_14[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 500] : memref<1500xf64>
              %9 = affine.load %alloca_13[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_0[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_11[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 250] : memref<1500xf64>
              %2 = affine.load %alloca_15[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 500] : memref<1500xf64>
              %5 = affine.load %alloca_14[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_3[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 625] : memref<1500xf64>
              %9 = affine.load %alloca_13[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_0[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_10[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_7[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_9[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_7[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_6[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_5[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_4[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %alloca_1[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %alloca_1[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_1[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_21 = memref.alloca() : memref<128xf64>
    %alloca_22 = memref.alloca() : memref<128xf64>
    %alloca_23 = memref.alloca() : memref<1500xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 6 {
        affine.for %arg7 = 0 to 125 {
          %0 = affine.load %arg2[%arg7 + %arg5 * 2250 + %arg6 * 125 + 750] : memref<?xf64>
          affine.store %0, %alloca_23[%arg7 + %arg5 * 750 + %arg6 * 125] : memref<1500xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_22[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %1 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
            affine.store %1, %alloca_21[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_24 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_25 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_26 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_27 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_28 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_29 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_30 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_31 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_32 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_33 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_34 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_35 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_36 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_37 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_38 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_39 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_40 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_22[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_40[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_22[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_39[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_39[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_38[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_40[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_37[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_40[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_36[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_38[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_35[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_37[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_34[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_36[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_33[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 750] : memref<1500xf64>
              %2 = affine.load %alloca_35[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 125] : memref<1500xf64>
              %5 = affine.load %alloca_34[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 250] : memref<1500xf64>
              %9 = affine.load %alloca_33[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_32[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 125] : memref<1500xf64>
              %2 = affine.load %alloca_35[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 375] : memref<1500xf64>
              %5 = affine.load %alloca_34[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 500] : memref<1500xf64>
              %9 = affine.load %alloca_33[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_0[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_31[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 250] : memref<1500xf64>
              %2 = affine.load %alloca_35[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 500] : memref<1500xf64>
              %5 = affine.load %alloca_34[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_23[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 625] : memref<1500xf64>
              %9 = affine.load %alloca_33[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_0[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_30[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_32[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_29[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_31[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_28[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_30[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_27[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_29[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_26[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_28[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_25[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_27[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_24[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_26[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_25[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_24[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %alloca_21[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %alloca_21[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_21[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_41 = memref.alloca() : memref<128xf64>
    %alloca_42 = memref.alloca() : memref<128xf64>
    %alloca_43 = memref.alloca() : memref<1500xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 6 {
        affine.for %arg7 = 0 to 125 {
          %0 = affine.load %arg2[%arg7 + %arg5 * 2250 + %arg6 * 125 + 1500] : memref<?xf64>
          affine.store %0, %alloca_43[%arg7 + %arg5 * 750 + %arg6 * 125] : memref<1500xf64>
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_42[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %1 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
            affine.store %1, %alloca_41[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_44 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_45 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_46 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_47 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_48 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_49 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_50 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_51 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_52 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_53 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_54 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_55 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_56 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_57 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_58 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_59 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_60 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_42[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_60[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_42[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_59[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_59[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_58[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_60[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_57[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_60[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_56[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_58[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_55[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_57[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_54[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_56[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_53[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 750] : memref<1500xf64>
              %2 = affine.load %alloca_55[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 125] : memref<1500xf64>
              %5 = affine.load %alloca_54[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 250] : memref<1500xf64>
              %9 = affine.load %alloca_53[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_52[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 125] : memref<1500xf64>
              %2 = affine.load %alloca_55[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 375] : memref<1500xf64>
              %5 = affine.load %alloca_54[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 500] : memref<1500xf64>
              %9 = affine.load %alloca_53[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_0[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_51[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 250] : memref<1500xf64>
              %2 = affine.load %alloca_55[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 500] : memref<1500xf64>
              %5 = affine.load %alloca_54[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %alloca_43[%arg6 * 25 + %arg9 + %arg5 * 750 + %arg7 * 5 + 625] : memref<1500xf64>
              %9 = affine.load %alloca_53[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %alloca_0[%arg9 + %arg8 * 5] : memref<20xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg10, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_50[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_52[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_49[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_51[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_48[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_50[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_47[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_49[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_46[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_48[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_0[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_45[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_47[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_44[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_46[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_45[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca_44[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %alloca_41[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %alloca_41[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_41[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_vector_convection_nl_apply_3d_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<750xf64>
    %alloca_0 = memref.alloca() : memref<2250xf64>
    %alloca_1 = memref.alloca() : memref<750xf64>
    %alloca_2 = memref.alloca() : memref<20xf64>
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 5 {
        %0 = affine.load %arg0[%arg5 + %arg6 * 4] : memref<?xf64>
        affine.store %0, %alloca_2[%arg6 + %arg5 * 5] : memref<20xf64>
      }
    }
    %alloca_3 = memref.alloca() : memref<750xf64>
    %alloca_4 = memref.alloca() : memref<250xf64>
    %alloca_5 = memref.alloca() : memref<128xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            affine.store %0, %alloca_5[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_6 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_7 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_7[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_7[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_6[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg6 * 25 + %arg8 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
          }
        }
      }
    }
    %alloca_8 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_9 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_10 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_11 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_12[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_9[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %alloca_4[%arg6 * 25 + %arg8 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
            affine.store %0, %alloca_1[%arg6 * 25 + %arg8 + %arg5 * 375 + %arg7 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 3 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.for %arg9 = 0 to 5 {
              %0 = affine.load %alloca_3[%arg5 * 375 + %arg6 * 125 + %arg7 + %arg9 * 25 + %arg8 * 5] : memref<750xf64>
              affine.store %0, %alloca_0[%arg5 * 1125 + %arg7 * 25 + %arg9 + %arg8 * 5 + %arg6 * 125] : memref<2250xf64>
            }
          }
        }
      }
    }
    %alloca_13 = memref.alloca() : memref<750xf64>
    %alloca_14 = memref.alloca() : memref<250xf64>
    %alloca_15 = memref.alloca() : memref<128xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_15[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_16 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_17 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_17[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_17[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_16[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_16[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_14[%arg6 * 25 + %arg8 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
          }
        }
      }
    }
    %alloca_18 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_19 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_20 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_21 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_22 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_22[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_21[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_21[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_22[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_22[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_18[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_20[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_19[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_18[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %alloca_14[%arg6 * 25 + %arg8 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
            affine.store %0, %alloca_1[%arg6 * 25 + %arg8 + %arg5 * 375 + %arg7 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 3 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.for %arg9 = 0 to 5 {
              %0 = affine.load %alloca_13[%arg5 * 375 + %arg6 * 125 + %arg7 + %arg9 * 25 + %arg8 * 5] : memref<750xf64>
              affine.store %0, %alloca_0[%arg5 * 1125 + %arg7 * 25 + %arg9 + %arg8 * 5 + %arg6 * 125 + 375] : memref<2250xf64>
            }
          }
        }
      }
    }
    %alloca_23 = memref.alloca() : memref<750xf64>
    %alloca_24 = memref.alloca() : memref<250xf64>
    %alloca_25 = memref.alloca() : memref<128xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_25[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_26 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_27 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_27[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_27[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_26[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_26[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_24[%arg6 * 25 + %arg8 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
          }
        }
      }
    }
    %alloca_28 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_29 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_30 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_31 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_32 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_32[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_25[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_31[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_31[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_30[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_32[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_29[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_32[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_28[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_30[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_23[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_29[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_23[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_28[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_23[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %alloca_24[%arg6 * 25 + %arg8 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
            affine.store %0, %alloca_1[%arg6 * 25 + %arg8 + %arg5 * 375 + %arg7 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 3 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.for %arg9 = 0 to 5 {
              %0 = affine.load %alloca_23[%arg5 * 375 + %arg6 * 125 + %arg7 + %arg9 * 25 + %arg8 * 5] : memref<750xf64>
              affine.store %0, %alloca_0[%arg5 * 1125 + %arg7 * 25 + %arg9 + %arg8 * 5 + %arg6 * 125 + 750] : memref<2250xf64>
            }
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 3 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.for %arg9 = 0 to 5 {
              %0 = affine.for %arg10 = 0 to 3 iter_args(%arg11 = %cst) -> (f64) {
                %1 = affine.load %alloca_1[%arg7 * 25 + %arg5 * 375 + %arg9 + %arg8 * 5 + %arg10 * 125] : memref<750xf64>
                %2 = affine.for %arg12 = 0 to 3 iter_args(%arg13 = %arg11) -> (f64) {
                  %3 = affine.load %alloca_0[%arg5 * 1125 + %arg6 * 375 + %arg7 * 25 + %arg9 + %arg8 * 5 + %arg12 * 125] : memref<2250xf64>
                  %4 = arith.mulf %1, %3 : f64
                  %5 = affine.load %arg2[%arg5 * 1125 + %arg10 * 375 + %arg7 * 25 + %arg9 + %arg8 * 5 + %arg12 * 125] : memref<?xf64>
                  %6 = arith.mulf %4, %5 : f64
                  %7 = arith.addf %arg13, %6 : f64
                  affine.yield %7 : f64
                }
                affine.yield %2 : f64
              }
              affine.store %0, %alloca[%arg7 * 25 + %arg5 * 375 + %arg9 + %arg8 * 5 + %arg6 * 125] : memref<750xf64>
            }
          }
        }
      }
    }
    %alloca_33 = memref.alloca() : memref<128xf64>
    %alloca_34 = memref.alloca() : memref<250xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %alloca[%arg6 + %arg5 * 375] : memref<750xf64>
        affine.store %0, %alloca_34[%arg6 + %arg5 * 125] : memref<250xf64>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            affine.store %0, %alloca_33[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_35 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_36 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_37 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_34[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg8 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_37[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_37[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_36[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_36[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_35[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_35[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_33[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_33[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_33[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_38 = memref.alloca() : memref<128xf64>
    %alloca_39 = memref.alloca() : memref<250xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %alloca[%arg6 + %arg5 * 375 + 125] : memref<750xf64>
        affine.store %0, %alloca_39[%arg6 + %arg5 * 125] : memref<250xf64>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_38[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_40 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_41 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_42 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_39[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg8 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_42[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_42[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_41[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_41[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_40[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_40[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_38[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_38[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_38[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_43 = memref.alloca() : memref<128xf64>
    %alloca_44 = memref.alloca() : memref<250xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %alloca[%arg6 + %arg5 * 375 + 250] : memref<750xf64>
        affine.store %0, %alloca_44[%arg6 + %arg5 * 125] : memref<250xf64>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_43[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_45 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_46 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_47 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_44[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg8 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_47[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_47[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_46[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_46[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_2[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_45[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_45[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_43[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_43[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_43[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_diffusion_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_2 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_3 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_4 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_8 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_9 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_10 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_11 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_12 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_13 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_14 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_15 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg7 * 64 + %arg11 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
              %2 = affine.load %arg0[%arg11 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_15[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg5[%arg7 * 64 + %arg11 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
              %2 = affine.load %arg1[%arg11 + %arg10 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_14[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_14[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg9 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_13[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg11 + %arg9 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_12[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg7, %arg8, %arg11, %arg10] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg9 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg7, %arg8, %arg9, %arg10] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_13[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_12[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg11 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %0 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg7, %arg11, %arg9, %arg10] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg11 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_8[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg9 * 5 + %arg7 * 750] : memref<?xf64>
              %2 = affine.load %alloca_10[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 125] : memref<?xf64>
              %5 = affine.load %alloca_9[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 250] : memref<?xf64>
              %9 = affine.load %alloca_8[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %arg3[%arg11 + %arg10 * 5] : memref<?xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg12, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_7[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 125] : memref<?xf64>
              %2 = affine.load %alloca_10[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 375] : memref<?xf64>
              %5 = affine.load %alloca_9[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 500] : memref<?xf64>
              %9 = affine.load %alloca_8[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %arg2[%arg11 + %arg10 * 5] : memref<?xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg12, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_6[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 250] : memref<?xf64>
              %2 = affine.load %alloca_10[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 500] : memref<?xf64>
              %5 = affine.load %alloca_9[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg4[%arg8 * 25 + %arg11 + %arg7 * 750 + %arg9 * 5 + 625] : memref<?xf64>
              %9 = affine.load %alloca_8[%arg7, %arg8, %arg9, %arg11] : memref<2x5x5x5xf64>
              %10 = arith.mulf %8, %9 : f64
              %11 = arith.addf %7, %10 : f64
              %12 = affine.load %arg2[%arg11 + %arg10 * 5] : memref<?xf64>
              %13 = arith.mulf %11, %12 : f64
              %14 = arith.addf %arg12, %13 : f64
              affine.yield %14 : f64
            }
            affine.store %0, %alloca_5[%arg7, %arg8, %arg9, %arg10] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_7[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_6[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg3[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg7, %arg8, %arg11, %arg10] : memref<2x5x5x4xf64>
              %2 = affine.load %arg2[%arg11 + %arg9 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg7, %arg8, %arg9, %arg10] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_4[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg11 + %arg8 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %2 = affine.load %arg2[%arg11 + %arg8 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg7, %arg11, %arg9, %arg10] : memref<2x5x4x4xf64>
              %2 = affine.load %arg3[%arg11 + %arg8 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg12, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 4 {
        affine.for %arg9 = 0 to 4 {
          affine.for %arg10 = 0 to 4 {
            %0 = affine.load %alloca_1[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_0[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x4xf64>
            %2 = arith.addf %0, %1 : f64
            %3 = affine.load %alloca[%arg7, %arg8, %arg9, %arg10] : memref<2x4x4x4xf64>
            %4 = arith.addf %2, %3 : f64
            %5 = affine.load %arg6[%arg7 * 64 + %arg10 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
            %6 = arith.addf %5, %4 : f64
            affine.store %6, %arg6[%arg7 * 64 + %arg10 + %arg8 * 16 + %arg9 * 4] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_discrete_gradient_apply_3d_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<750xf64>
    %alloca_0 = memref.alloca() : memref<750xf64>
    %alloca_1 = memref.alloca() : memref<20xf64>
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 5 {
        %0 = affine.load %arg0[%arg5 + %arg6 * 4] : memref<?xf64>
        affine.store %0, %alloca_1[%arg6 + %arg5 * 5] : memref<20xf64>
      }
    }
    %alloca_2 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_4 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_5 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_6 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg3[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
              %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_6[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg3[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
              %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_5[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_5[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_4[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_6[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_6[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_4[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 125] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg9 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg5 * 375 + %arg6 + %arg8 * 25 + %arg7 * 5 + 250] : memref<750xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 3 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.for %arg9 = 0 to 5 {
              %0 = affine.for %arg10 = 0 to 3 iter_args(%arg11 = %cst) -> (f64) {
                %1 = affine.load %alloca_0[%arg5 * 375 + %arg10 * 125 + %arg7 + %arg9 * 25 + %arg8 * 5] : memref<750xf64>
                %2 = affine.load %arg2[%arg5 * 1125 + %arg6 * 375 + %arg7 * 25 + %arg9 + %arg8 * 5 + %arg10 * 125] : memref<?xf64>
                %3 = arith.mulf %1, %2 : f64
                %4 = arith.addf %arg11, %3 : f64
                affine.yield %4 : f64
              }
              affine.store %0, %alloca[%arg7 * 25 + %arg5 * 375 + %arg9 + %arg8 * 5 + %arg6 * 125] : memref<750xf64>
            }
          }
        }
      }
    }
    %alloca_7 = memref.alloca() : memref<128xf64>
    %alloca_8 = memref.alloca() : memref<250xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %alloca[%arg6 + %arg5 * 375] : memref<750xf64>
        affine.store %0, %alloca_8[%arg6 + %arg5 * 125] : memref<250xf64>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            affine.store %0, %alloca_7[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_9 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_10 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_11 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_8[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg8 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_11[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_11[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_10[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_10[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_9[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_9[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_7[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_7[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_7[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          }
        }
      }
    }
    %alloca_12 = memref.alloca() : memref<128xf64>
    %alloca_13 = memref.alloca() : memref<250xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %alloca[%arg6 + %arg5 * 375 + 125] : memref<750xf64>
        affine.store %0, %alloca_13[%arg6 + %arg5 * 125] : memref<250xf64>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
            affine.store %0, %alloca_12[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_14 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_15 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_16 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_13[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg8 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_16[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_16[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_15[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_15[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_14[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_14[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_12[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_12[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_12[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 64] : memref<?xf64>
          }
        }
      }
    }
    %alloca_17 = memref.alloca() : memref<128xf64>
    %alloca_18 = memref.alloca() : memref<250xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %alloca[%arg6 + %arg5 * 375 + 250] : memref<750xf64>
        affine.store %0, %alloca_18[%arg6 + %arg5 * 125] : memref<250xf64>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
            affine.store %0, %alloca_17[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    %alloca_19 = memref.alloca() : memref<2x4x4x4xf64>
    %alloca_20 = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_21 = memref.alloca() : memref<2x5x5x4xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_18[%arg6 * 25 + %arg9 + %arg7 * 5 + %arg5 * 125] : memref<250xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg8 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_21[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_21[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg7 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_20[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %alloca_20[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %2 = affine.load %alloca_1[%arg9 + %arg6 * 5] : memref<20xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_19[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_19[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x4xf64>
            %1 = affine.load %alloca_17[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %alloca_17[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %alloca_17[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<128xf64>
            affine.store %0, %arg4[%arg5 * 192 + %arg8 + %arg6 * 16 + %arg7 * 4 + 128] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_discrete_divergence_apply_3d_direct(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x5x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %arg10) -> (f64) {
                %2 = affine.load %arg0[%arg11 + %arg6 * 4] : memref<?xf64>
                %3 = affine.load %arg2[%arg6 * 25 + %arg5 * 1125 + %arg8 + %arg7 * 5 + %arg9 * 375] : memref<?xf64>
                %4 = affine.load %arg2[%arg5 * 1125 + %arg9 * 375 + %arg8 + %arg6 * 25 + %arg7 * 5 + 125] : memref<?xf64>
                %5 = affine.load %arg1[%arg11 + %arg6 * 4] : memref<?xf64>
                %6 = affine.load %arg2[%arg5 * 1125 + %arg9 * 375 + %arg8 + %arg6 * 25 + %arg7 * 5 + 250] : memref<?xf64>
                %7 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %arg12) -> (f64) {
                  %8 = affine.load %arg0[%arg13 + %arg7 * 4] : memref<?xf64>
                  %9 = affine.load %arg1[%arg13 + %arg7 * 4] : memref<?xf64>
                  %10 = affine.for %arg15 = 0 to 4 iter_args(%arg16 = %arg14) -> (f64) {
                    %11 = affine.load %arg3[%arg5 * 192 + %arg9 * 64 + %arg15 + %arg11 * 16 + %arg13 * 4] : memref<?xf64>
                    %12 = affine.load %arg1[%arg15 + %arg8 * 4] : memref<?xf64>
                    %13 = arith.mulf %11, %12 : f64
                    %14 = arith.mulf %13, %8 : f64
                    %15 = arith.mulf %14, %2 : f64
                    %16 = arith.mulf %15, %3 : f64
                    %17 = arith.addf %arg16, %16 : f64
                    %18 = affine.load %arg0[%arg15 + %arg8 * 4] : memref<?xf64>
                    %19 = arith.mulf %11, %18 : f64
                    %20 = arith.mulf %19, %9 : f64
                    %21 = arith.mulf %20, %2 : f64
                    %22 = arith.mulf %21, %4 : f64
                    %23 = arith.addf %17, %22 : f64
                    %24 = arith.mulf %19, %8 : f64
                    %25 = arith.mulf %24, %5 : f64
                    %26 = arith.mulf %25, %6 : f64
                    %27 = arith.addf %23, %26 : f64
                    affine.yield %27 : f64
                  }
                  affine.yield %10 : f64
                }
                affine.yield %7 : f64
              }
              affine.yield %1 : f64
            }
            affine.store %0, %alloca[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %3 = affine.load %arg0[%arg6 + %arg9 * 4] : memref<?xf64>
              %4 = affine.for %arg11 = 0 to 5 iter_args(%arg12 = %arg10) -> (f64) {
                %5 = affine.load %arg0[%arg7 + %arg11 * 4] : memref<?xf64>
                %6 = affine.for %arg13 = 0 to 5 iter_args(%arg14 = %arg12) -> (f64) {
                  %7 = affine.load %alloca[%arg5, %arg9, %arg11, %arg13] : memref<2x5x5x5xf64>
                  %8 = affine.load %arg0[%arg8 + %arg13 * 4] : memref<?xf64>
                  %9 = arith.mulf %7, %8 : f64
                  %10 = arith.mulf %9, %5 : f64
                  %11 = arith.mulf %10, %3 : f64
                  %12 = arith.addf %arg14, %11 : f64
                  affine.yield %12 : f64
                }
                affine.yield %6 : f64
              }
              affine.yield %4 : f64
            }
            %1 = affine.load %arg4[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg4[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_interp_grad_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_0 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_2 = memref.alloca() : memref<2x4x4x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg4 * 64 + %arg8 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
              %2 = affine.load %arg1[%arg8 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg4, %arg5, %arg6, %arg7] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg4 * 64 + %arg8 + %arg5 * 16 + %arg6 * 4] : memref<?xf64>
              %2 = affine.load %arg2[%arg8 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg4, %arg5, %arg6, %arg7] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca_2[%arg4, %arg5, %arg8, %arg7] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg8 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg4, %arg5, %arg6, %arg7] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg4, %arg5, %arg8, %arg7] : memref<2x4x4x5xf64>
              %2 = affine.load %arg2[%arg8 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg4, %arg5, %arg6, %arg7] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca_3[%arg4, %arg5, %arg8, %arg7] : memref<2x4x4x5xf64>
              %2 = affine.load %arg1[%arg8 + %arg6 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg4, %arg5, %arg6, %arg7] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca_1[%arg4, %arg8, %arg6, %arg7] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg8 + %arg5 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %arg3[%arg4 * 375 + %arg5 + %arg7 * 25 + %arg6 * 5] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca_0[%arg4, %arg8, %arg6, %arg7] : memref<2x4x5x5xf64>
              %2 = affine.load %arg1[%arg8 + %arg5 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %arg3[%arg4 * 375 + %arg5 + %arg7 * 25 + %arg6 * 5 + 125] : memref<?xf64>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          affine.for %arg7 = 0 to 5 {
            %0 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %cst) -> (f64) {
              %1 = affine.load %alloca[%arg4, %arg8, %arg6, %arg7] : memref<2x4x5x5xf64>
              %2 = affine.load %arg2[%arg8 + %arg5 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg9, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %arg3[%arg4 * 375 + %arg5 + %arg7 * 25 + %arg6 * 5 + 250] : memref<?xf64>
          }
        }
      }
    }
    return
  }
  func.func @mfem_pa_mass_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x5x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x5x5x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x5x5xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x4x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
              %2 = affine.load %arg3[%arg5 * 64 + %arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_3[%arg5, %arg6, %arg7, %arg8] : memref<2x4x4x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
              %2 = affine.load %alloca_3[%arg5, %arg6, %arg9, %arg8] : memref<2x4x4x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_2[%arg5, %arg6, %arg7, %arg8] : memref<2x4x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg0[%arg9 + %arg6 * 4] : memref<?xf64>
              %2 = affine.load %alloca_2[%arg5, %arg9, %arg7, %arg8] : memref<2x4x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_1[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %arg2[%arg5 * 125 + %arg8 + %arg6 * 25 + %arg7 * 5] : memref<?xf64>
            %1 = affine.load %alloca_1[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_1[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x5xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg1[%arg9 + %arg8 * 5] : memref<?xf64>
              %2 = affine.load %alloca_1[%arg5, %arg6, %arg7, %arg9] : memref<2x5x5x5xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca_0[%arg5, %arg6, %arg7, %arg8] : memref<2x5x5x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %1 = affine.load %arg1[%arg9 + %arg7 * 5] : memref<?xf64>
              %2 = affine.load %alloca_0[%arg5, %arg6, %arg9, %arg8] : memref<2x5x5x4xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %arg10, %3 : f64
              affine.yield %4 : f64
            }
            affine.store %0, %alloca[%arg5, %arg6, %arg7, %arg8] : memref<2x5x4x4xf64>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
              %3 = affine.load %arg1[%arg9 + %arg6 * 5] : memref<?xf64>
              %4 = affine.load %alloca[%arg5, %arg9, %arg7, %arg8] : memref<2x5x4x4xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.addf %arg10, %5 : f64
              affine.yield %6 : f64
            }
            %1 = affine.load %arg4[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            %2 = arith.addf %1, %0 : f64
            affine.store %2, %arg4[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
