module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_curlcurl_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<3x2xf64>
    %alloca_0 = memref.alloca() : memref<3x4xf64>
    %alloca_1 = memref.alloca() : memref<3x4xf64>
    %alloca_2 = memref.alloca() : memref<3x2xf64>
    %alloca_3 = memref.alloca() : memref<3x4xf64>
    %alloca_4 = memref.alloca() : memref<3x4xf64>
    %alloca_5 = memref.alloca() : memref<3x2xf64>
    %alloca_6 = memref.alloca() : memref<4x3xf64>
    %alloca_7 = memref.alloca() : memref<4x3xf64>
    %alloca_8 = memref.alloca() : memref<5xf64>
    %alloca_9 = memref.alloca() : memref<5x5x2xf64>
    %alloca_10 = memref.alloca() : memref<5xf64>
    %alloca_11 = memref.alloca() : memref<5x5x2xf64>
    %alloca_12 = memref.alloca() : memref<5xf64>
    %alloca_13 = memref.alloca() : memref<5x5x2xf64>
    %alloca_14 = memref.alloca() : memref<5x5x5x3xf64>
    affine.for %arg9 = 0 to 2 {
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            affine.for %arg13 = 0 to 3 {
              affine.store %cst, %alloca_14[%arg10, %arg11, %arg12, %arg13] : memref<5x5x5x3xf64>
            }
          }
        }
      }
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            affine.store %cst, %alloca_13[%arg11, %arg12, 1] : memref<5x5x2xf64>
            affine.store %cst, %alloca_13[%arg11, %arg12, 0] : memref<5x5x2xf64>
          }
        }
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 5 {
            affine.store %cst, %alloca_12[%arg12] : memref<5xf64>
          }
          affine.for %arg12 = 0 to 3 {
            %0 = affine.load %arg7[%arg10 * 12 + %arg12 + %arg11 * 3 + %arg9 * 144] : memref<?xf64>
            affine.for %arg13 = 0 to 5 {
              %1 = affine.load %arg0[%arg12 + %arg13 * 3] : memref<?xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_12[%arg13] : memref<5xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_12[%arg13] : memref<5xf64>
            }
          }
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %arg4[%arg11 + %arg12 * 4] : memref<?xf64>
            %1 = affine.load %arg1[%arg11 + %arg12 * 4] : memref<?xf64>
            affine.for %arg13 = 0 to 5 {
              %2 = affine.load %alloca_12[%arg13] : memref<5xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_13[%arg12, %arg13, 0] : memref<5x5x2xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_13[%arg12, %arg13, 0] : memref<5x5x2xf64>
              %6 = arith.mulf %2, %1 : f64
              %7 = affine.load %alloca_13[%arg12, %arg13, 1] : memref<5x5x2xf64>
              %8 = arith.addf %7, %6 : f64
              affine.store %8, %alloca_13[%arg12, %arg13, 1] : memref<5x5x2xf64>
            }
          }
        }
        affine.for %arg11 = 0 to 5 {
          %0 = affine.load %arg4[%arg10 + %arg11 * 4] : memref<?xf64>
          %1 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
          affine.for %arg12 = 0 to 5 {
            affine.for %arg13 = 0 to 5 {
              %2 = affine.load %alloca_13[%arg12, %arg13, 1] : memref<5x5x2xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_14[%arg11, %arg12, %arg13, 1] : memref<5x5x5x3xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_14[%arg11, %arg12, %arg13, 1] : memref<5x5x5x3xf64>
              %6 = affine.load %alloca_13[%arg12, %arg13, 0] : memref<5x5x2xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_14[%arg11, %arg12, %arg13, 2] : memref<5x5x5x3xf64>
              %9 = arith.subf %8, %7 : f64
              affine.store %9, %alloca_14[%arg11, %arg12, %arg13, 2] : memref<5x5x5x3xf64>
            }
          }
        }
      }
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            affine.store %cst, %alloca_11[%arg11, %arg12, 1] : memref<5x5x2xf64>
            affine.store %cst, %alloca_11[%arg11, %arg12, 0] : memref<5x5x2xf64>
          }
        }
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 5 {
            affine.store %cst, %alloca_10[%arg12] : memref<5xf64>
          }
          affine.for %arg12 = 0 to 3 {
            %0 = affine.load %arg7[%arg10 * 12 + %arg11 + %arg12 * 4 + %arg9 * 144 + 48] : memref<?xf64>
            affine.for %arg13 = 0 to 5 {
              %1 = affine.load %arg0[%arg12 + %arg13 * 3] : memref<?xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_10[%arg13] : memref<5xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_10[%arg13] : memref<5xf64>
            }
          }
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %arg4[%arg11 + %arg12 * 4] : memref<?xf64>
            %1 = affine.load %arg1[%arg11 + %arg12 * 4] : memref<?xf64>
            affine.for %arg13 = 0 to 5 {
              %2 = affine.load %alloca_10[%arg13] : memref<5xf64>
              %3 = arith.mulf %0, %2 : f64
              %4 = affine.load %alloca_11[%arg13, %arg12, 0] : memref<5x5x2xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_11[%arg13, %arg12, 0] : memref<5x5x2xf64>
              %6 = arith.mulf %1, %2 : f64
              %7 = affine.load %alloca_11[%arg13, %arg12, 1] : memref<5x5x2xf64>
              %8 = arith.addf %7, %6 : f64
              affine.store %8, %alloca_11[%arg13, %arg12, 1] : memref<5x5x2xf64>
            }
          }
        }
        affine.for %arg11 = 0 to 5 {
          %0 = affine.load %arg4[%arg10 + %arg11 * 4] : memref<?xf64>
          %1 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
          affine.for %arg12 = 0 to 5 {
            affine.for %arg13 = 0 to 5 {
              %2 = affine.load %alloca_11[%arg12, %arg13, 1] : memref<5x5x2xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_14[%arg11, %arg12, %arg13, 0] : memref<5x5x5x3xf64>
              %5 = arith.subf %4, %3 : f64
              affine.store %5, %alloca_14[%arg11, %arg12, %arg13, 0] : memref<5x5x5x3xf64>
              %6 = affine.load %alloca_11[%arg12, %arg13, 0] : memref<5x5x2xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_14[%arg11, %arg12, %arg13, 2] : memref<5x5x5x3xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_14[%arg11, %arg12, %arg13, 2] : memref<5x5x5x3xf64>
            }
          }
        }
      }
      affine.for %arg10 = 0 to 4 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            affine.store %cst, %alloca_9[%arg11, %arg12, 1] : memref<5x5x2xf64>
            affine.store %cst, %alloca_9[%arg11, %arg12, 0] : memref<5x5x2xf64>
          }
        }
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 5 {
            affine.store %cst, %alloca_8[%arg12] : memref<5xf64>
          }
          affine.for %arg12 = 0 to 3 {
            %0 = affine.load %arg7[%arg12 * 16 + %arg10 + %arg11 * 4 + %arg9 * 144 + 96] : memref<?xf64>
            affine.for %arg13 = 0 to 5 {
              %1 = affine.load %arg0[%arg12 + %arg13 * 3] : memref<?xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_8[%arg13] : memref<5xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_8[%arg13] : memref<5xf64>
            }
          }
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %arg1[%arg11 + %arg12 * 4] : memref<?xf64>
            %1 = affine.load %arg4[%arg11 + %arg12 * 4] : memref<?xf64>
            affine.for %arg13 = 0 to 5 {
              %2 = affine.load %alloca_8[%arg13] : memref<5xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_9[%arg13, %arg12, 0] : memref<5x5x2xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_9[%arg13, %arg12, 0] : memref<5x5x2xf64>
              %6 = arith.mulf %2, %1 : f64
              %7 = affine.load %alloca_9[%arg13, %arg12, 1] : memref<5x5x2xf64>
              %8 = arith.addf %7, %6 : f64
              affine.store %8, %alloca_9[%arg13, %arg12, 1] : memref<5x5x2xf64>
            }
          }
        }
        affine.for %arg11 = 0 to 5 {
          %0 = affine.load %arg1[%arg10 + %arg11 * 4] : memref<?xf64>
          %1 = affine.load %arg4[%arg10 + %arg11 * 4] : memref<?xf64>
          affine.for %arg12 = 0 to 5 {
            affine.for %arg13 = 0 to 5 {
              %2 = affine.load %alloca_9[%arg13, %arg12, 1] : memref<5x5x2xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_14[%arg13, %arg12, %arg11, 0] : memref<5x5x5x3xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_14[%arg13, %arg12, %arg11, 0] : memref<5x5x5x3xf64>
              %6 = affine.load %alloca_9[%arg13, %arg12, 0] : memref<5x5x2xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_14[%arg13, %arg12, %arg11, 1] : memref<5x5x5x3xf64>
              %9 = arith.subf %8, %7 : f64
              affine.store %9, %alloca_14[%arg13, %arg12, %arg11, 1] : memref<5x5x5x3xf64>
            }
          }
        }
      }
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %alloca_14[%arg10, %arg11, %arg12, 0] : memref<5x5x5x3xf64>
            %1 = affine.load %alloca_14[%arg10, %arg11, %arg12, 1] : memref<5x5x5x3xf64>
            %2 = affine.load %alloca_14[%arg10, %arg11, %arg12, 2] : memref<5x5x5x3xf64>
            %3 = affine.load %arg6[%arg9 * 750 + %arg12 + %arg10 * 25 + %arg11 * 5] : memref<?xf64>
            %4 = affine.load %arg6[%arg9 * 750 + %arg12 + %arg10 * 25 + %arg11 * 5 + 125] : memref<?xf64>
            %5 = affine.load %arg6[%arg9 * 750 + %arg12 + %arg10 * 25 + %arg11 * 5 + 250] : memref<?xf64>
            %6 = affine.load %arg6[%arg9 * 750 + %arg12 + %arg10 * 25 + %arg11 * 5 + 375] : memref<?xf64>
            %7 = affine.load %arg6[%arg9 * 750 + %arg12 + %arg10 * 25 + %arg11 * 5 + 500] : memref<?xf64>
            %8 = affine.load %arg6[%arg9 * 750 + %arg12 + %arg10 * 25 + %arg11 * 5 + 625] : memref<?xf64>
            %9 = arith.mulf %3, %0 : f64
            %10 = arith.mulf %4, %1 : f64
            %11 = arith.addf %9, %10 : f64
            %12 = arith.mulf %5, %2 : f64
            %13 = arith.addf %11, %12 : f64
            affine.store %13, %alloca_14[%arg10, %arg11, %arg12, 0] : memref<5x5x5x3xf64>
            %14 = arith.mulf %4, %0 : f64
            %15 = arith.mulf %6, %1 : f64
            %16 = arith.addf %14, %15 : f64
            %17 = arith.mulf %7, %2 : f64
            %18 = arith.addf %16, %17 : f64
            affine.store %18, %alloca_14[%arg10, %arg11, %arg12, 1] : memref<5x5x5x3xf64>
            %19 = arith.mulf %5, %0 : f64
            %20 = arith.mulf %7, %1 : f64
            %21 = arith.addf %19, %20 : f64
            %22 = arith.mulf %8, %2 : f64
            %23 = arith.addf %21, %22 : f64
            affine.store %23, %alloca_14[%arg10, %arg11, %arg12, 2] : memref<5x5x5x3xf64>
          }
        }
      }
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            affine.store %cst, %alloca_6[%arg11, %arg12] : memref<4x3xf64>
            affine.store %cst, %alloca_7[%arg11, %arg12] : memref<4x3xf64>
          }
        }
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 3 {
            affine.store %cst, %alloca_5[%arg12, 1] : memref<3x2xf64>
            affine.store %cst, %alloca_5[%arg12, 0] : memref<3x2xf64>
          }
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %alloca_14[%arg10, %arg11, %arg12, 1] : memref<5x5x5x3xf64>
            %1 = affine.load %alloca_14[%arg10, %arg11, %arg12, 2] : memref<5x5x5x3xf64>
            affine.for %arg13 = 0 to 3 {
              %2 = affine.load %arg2[%arg12 + %arg13 * 5] : memref<?xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_5[%arg13, 0] : memref<3x2xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_5[%arg13, 0] : memref<3x2xf64>
              %6 = affine.load %arg2[%arg12 + %arg13 * 5] : memref<?xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_5[%arg13, 1] : memref<3x2xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_5[%arg13, 1] : memref<3x2xf64>
            }
          }
          affine.for %arg12 = 0 to 4 {
            %0 = affine.load %arg3[%arg11 + %arg12 * 5] : memref<?xf64>
            %1 = affine.load %arg5[%arg11 + %arg12 * 5] : memref<?xf64>
            affine.for %arg13 = 0 to 3 {
              %2 = affine.load %alloca_5[%arg13, 0] : memref<3x2xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_6[%arg12, %arg13] : memref<4x3xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_6[%arg12, %arg13] : memref<4x3xf64>
              %6 = affine.load %alloca_5[%arg13, 1] : memref<3x2xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_7[%arg12, %arg13] : memref<4x3xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_7[%arg12, %arg13] : memref<4x3xf64>
            }
          }
        }
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            affine.for %arg13 = 0 to 3 {
              %0 = affine.load %alloca_6[%arg12, %arg13] : memref<4x3xf64>
              %1 = affine.load %arg5[%arg10 + %arg11 * 5] : memref<?xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_7[%arg12, %arg13] : memref<4x3xf64>
              %4 = affine.load %arg3[%arg10 + %arg11 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.subf %2, %5 : f64
              %7 = affine.load %arg8[%arg11 * 12 + %arg13 + %arg12 * 3 + %arg9 * 144] : memref<?xf64>
              %8 = arith.addf %7, %6 : f64
              affine.store %8, %arg8[%arg11 * 12 + %arg13 + %arg12 * 3 + %arg9 * 144] : memref<?xf64>
            }
          }
        }
      }
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            affine.store %cst, %alloca_3[%arg11, %arg12] : memref<3x4xf64>
            affine.store %cst, %alloca_4[%arg11, %arg12] : memref<3x4xf64>
          }
        }
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 3 {
            affine.store %cst, %alloca_2[%arg12, 1] : memref<3x2xf64>
            affine.store %cst, %alloca_2[%arg12, 0] : memref<3x2xf64>
          }
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %alloca_14[%arg10, %arg12, %arg11, 2] : memref<5x5x5x3xf64>
            %1 = affine.load %alloca_14[%arg10, %arg12, %arg11, 0] : memref<5x5x5x3xf64>
            affine.for %arg13 = 0 to 3 {
              %2 = affine.load %arg2[%arg12 + %arg13 * 5] : memref<?xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_2[%arg13, 0] : memref<3x2xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_2[%arg13, 0] : memref<3x2xf64>
              %6 = affine.load %arg2[%arg12 + %arg13 * 5] : memref<?xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_2[%arg13, 1] : memref<3x2xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_2[%arg13, 1] : memref<3x2xf64>
            }
          }
          affine.for %arg12 = 0 to 4 {
            %0 = affine.load %arg5[%arg11 + %arg12 * 5] : memref<?xf64>
            %1 = affine.load %arg3[%arg11 + %arg12 * 5] : memref<?xf64>
            affine.for %arg13 = 0 to 3 {
              %2 = affine.load %alloca_2[%arg13, 0] : memref<3x2xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca_4[%arg13, %arg12] : memref<3x4xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_4[%arg13, %arg12] : memref<3x4xf64>
              %6 = affine.load %alloca_2[%arg13, 1] : memref<3x2xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca_3[%arg13, %arg12] : memref<3x4xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_3[%arg13, %arg12] : memref<3x4xf64>
            }
          }
        }
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 3 {
            affine.for %arg13 = 0 to 4 {
              %0 = affine.load %alloca_3[%arg12, %arg13] : memref<3x4xf64>
              %1 = arith.negf %0 : f64
              %2 = affine.load %arg5[%arg10 + %arg11 * 5] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = affine.load %alloca_4[%arg12, %arg13] : memref<3x4xf64>
              %5 = affine.load %arg3[%arg10 + %arg11 * 5] : memref<?xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.addf %3, %6 : f64
              %8 = affine.load %arg8[%arg11 * 12 + %arg13 + %arg12 * 4 + %arg9 * 144 + 48] : memref<?xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %arg8[%arg11 * 12 + %arg13 + %arg12 * 4 + %arg9 * 144 + 48] : memref<?xf64>
            }
          }
        }
      }
      affine.for %arg10 = 0 to 5 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 4 {
            affine.store %cst, %alloca_0[%arg11, %arg12] : memref<3x4xf64>
            affine.store %cst, %alloca_1[%arg11, %arg12] : memref<3x4xf64>
          }
        }
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 3 {
            affine.store %cst, %alloca[%arg12, 1] : memref<3x2xf64>
            affine.store %cst, %alloca[%arg12, 0] : memref<3x2xf64>
          }
          affine.for %arg12 = 0 to 5 {
            %0 = affine.load %alloca_14[%arg12, %arg11, %arg10, 0] : memref<5x5x5x3xf64>
            %1 = affine.load %alloca_14[%arg12, %arg11, %arg10, 1] : memref<5x5x5x3xf64>
            affine.for %arg13 = 0 to 3 {
              %2 = affine.load %arg2[%arg12 + %arg13 * 5] : memref<?xf64>
              %3 = arith.mulf %2, %0 : f64
              %4 = affine.load %alloca[%arg13, 0] : memref<3x2xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca[%arg13, 0] : memref<3x2xf64>
              %6 = affine.load %arg2[%arg12 + %arg13 * 5] : memref<?xf64>
              %7 = arith.mulf %6, %1 : f64
              %8 = affine.load %alloca[%arg13, 1] : memref<3x2xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca[%arg13, 1] : memref<3x2xf64>
            }
          }
          affine.for %arg12 = 0 to 4 {
            %0 = affine.load %arg3[%arg11 + %arg12 * 5] : memref<?xf64>
            %1 = affine.load %arg5[%arg11 + %arg12 * 5] : memref<?xf64>
            affine.for %arg13 = 0 to 3 {
              %2 = affine.load %alloca[%arg13, 1] : memref<3x2xf64>
              %3 = arith.mulf %0, %2 : f64
              %4 = affine.load %alloca_1[%arg13, %arg12] : memref<3x4xf64>
              %5 = arith.addf %4, %3 : f64
              affine.store %5, %alloca_1[%arg13, %arg12] : memref<3x4xf64>
              %6 = affine.load %alloca[%arg13, 0] : memref<3x2xf64>
              %7 = arith.mulf %1, %6 : f64
              %8 = affine.load %alloca_0[%arg13, %arg12] : memref<3x4xf64>
              %9 = arith.addf %8, %7 : f64
              affine.store %9, %alloca_0[%arg13, %arg12] : memref<3x4xf64>
            }
          }
        }
        affine.for %arg11 = 0 to 4 {
          affine.for %arg12 = 0 to 4 {
            affine.for %arg13 = 0 to 3 {
              %0 = affine.load %alloca_0[%arg13, %arg12] : memref<3x4xf64>
              %1 = affine.load %arg3[%arg10 + %arg11 * 5] : memref<?xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_1[%arg13, %arg12] : memref<3x4xf64>
              %4 = affine.load %arg5[%arg10 + %arg11 * 5] : memref<?xf64>
              %5 = arith.mulf %3, %4 : f64
              %6 = arith.subf %2, %5 : f64
              %7 = affine.load %arg8[%arg13 * 16 + %arg11 + %arg12 * 4 + %arg9 * 144 + 96] : memref<?xf64>
              %8 = arith.addf %7, %6 : f64
              affine.store %8, %arg8[%arg13 * 16 + %arg11 + %arg12 * 4 + %arg9 * 144 + 96] : memref<?xf64>
            }
          }
        }
      }
    }
    return
  }
}
