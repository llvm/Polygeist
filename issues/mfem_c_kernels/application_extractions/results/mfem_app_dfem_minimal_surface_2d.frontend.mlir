module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_dfem_minimal_surface_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<100xf64>
    %alloca_1 = memref.alloca() : memref<100xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x5xf64>
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %arg2[%arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            %2 = affine.load %arg0[%arg9 + %arg8 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_3[%arg6, %arg7, %arg8] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %arg2[%arg9 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            %2 = affine.load %arg1[%arg9 + %arg8 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_2[%arg6, %arg7, %arg8] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_2[%arg6, %arg9, %arg8] : memref<2x4x5xf64>
            %2 = affine.load %arg0[%arg9 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_1[%arg7 + %arg6 * 50 + %arg8 * 5] : memref<100xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 5 {
          %0 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_3[%arg6, %arg9, %arg8] : memref<2x4x5xf64>
            %2 = affine.load %arg1[%arg9 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_1[%arg7 + %arg6 * 50 + %arg8 * 5 + 25] : memref<100xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 5 {
      affine.for %arg7 = 0 to 5 {
        %0 = affine.load %alloca_1[%arg7 + %arg6 * 5] : memref<100xf64>
        %1 = affine.load %alloca_1[%arg7 + %arg6 * 5 + 25] : memref<100xf64>
        %2 = affine.load %arg3[%arg7 * 4 + %arg6 * 20] : memref<?xf64>
        %3 = affine.load %arg3[%arg7 * 4 + %arg6 * 20 + 1] : memref<?xf64>
        %4 = affine.load %arg3[%arg7 * 4 + %arg6 * 20 + 2] : memref<?xf64>
        %5 = affine.load %arg3[%arg7 * 4 + %arg6 * 20 + 3] : memref<?xf64>
        %6 = arith.mulf %2, %5 : f64
        %7 = arith.mulf %3, %4 : f64
        %8 = arith.subf %6, %7 : f64
        %9 = arith.divf %5, %8 : f64
        %10 = arith.negf %3 : f64
        %11 = arith.divf %10, %8 : f64
        %12 = arith.negf %4 : f64
        %13 = arith.divf %12, %8 : f64
        %14 = arith.divf %2, %8 : f64
        %15 = arith.mulf %0, %9 : f64
        %16 = arith.mulf %1, %13 : f64
        %17 = arith.addf %15, %16 : f64
        %18 = arith.mulf %0, %11 : f64
        %19 = arith.mulf %1, %14 : f64
        %20 = arith.addf %18, %19 : f64
        %21 = arith.mulf %17, %17 : f64
        %22 = arith.addf %21, %cst_0 : f64
        %23 = arith.mulf %20, %20 : f64
        %24 = arith.addf %22, %23 : f64
        %25 = math.sqrt %24 : f64
        %26 = arith.divf %cst_0, %25 : f64
        %27 = arith.mulf %26, %8 : f64
        %28 = affine.load %arg4[%arg7 + %arg6 * 5] : memref<?xf64>
        %29 = arith.mulf %27, %28 : f64
        %30 = arith.mulf %17, %9 : f64
        %31 = arith.mulf %20, %11 : f64
        %32 = arith.addf %30, %31 : f64
        %33 = arith.mulf %29, %32 : f64
        affine.store %33, %alloca[%arg7 + %arg6 * 5] : memref<100xf64>
        %34 = arith.mulf %17, %13 : f64
        %35 = arith.mulf %20, %14 : f64
        %36 = arith.addf %34, %35 : f64
        %37 = arith.mulf %29, %36 : f64
        affine.store %37, %alloca[%arg7 + %arg6 * 5 + 25] : memref<100xf64>
        affine.store %cst, %alloca[%arg7 + %arg6 * 5 + 50] : memref<100xf64>
        affine.store %cst, %alloca[%arg7 + %arg6 * 5 + 75] : memref<100xf64>
      }
    }
    %alloca_4 = memref.alloca() : memref<2x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4xf64>
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca[%arg7 + %arg6 * 50 + %arg9 * 5] : memref<100xf64>
            %2 = affine.load %arg1[%arg8 + %arg9 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_7[%arg6, %arg7, %arg8] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 5 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca[%arg7 + %arg6 * 50 + %arg9 * 5 + 25] : memref<100xf64>
            %2 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_6[%arg6, %arg7, %arg8] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_7[%arg6, %arg9, %arg8] : memref<2x5x4xf64>
            %2 = affine.load %arg0[%arg7 + %arg9 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_5[%arg6, %arg7, %arg8] : memref<2x4x4xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.for %arg9 = 0 to 5 iter_args(%arg10 = %cst) -> (f64) {
            %1 = affine.load %alloca_6[%arg6, %arg9, %arg8] : memref<2x5x4xf64>
            %2 = affine.load %arg1[%arg7 + %arg9 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg10, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_4[%arg6, %arg7, %arg8] : memref<2x4x4xf64>
        }
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 4 {
          %0 = affine.load %alloca_5[%arg6, %arg7, %arg8] : memref<2x4x4xf64>
          %1 = affine.load %alloca_4[%arg6, %arg7, %arg8] : memref<2x4x4xf64>
          %2 = arith.addf %0, %1 : f64
          %3 = affine.load %arg5[%arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg5[%arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
        }
      }
    }
    return
  }
  func.func @mfem_interp_grad_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x5xf64>
    %alloca_0 = memref.alloca() : memref<2x4x5xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          %0 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %arg0[%arg7 + %arg4 * 16 + %arg5 * 4] : memref<?xf64>
            %2 = affine.load %arg1[%arg7 + %arg6 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_0[%arg4, %arg5, %arg6] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 5 {
          %0 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %arg0[%arg7 + %arg4 * 16 + %arg5 * 4] : memref<?xf64>
            %2 = affine.load %arg2[%arg7 + %arg6 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca[%arg4, %arg5, %arg6] : memref<2x4x5xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          %0 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %alloca[%arg4, %arg7, %arg6] : memref<2x4x5xf64>
            %2 = affine.load %arg1[%arg7 + %arg5 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %arg3[%arg5 + %arg4 * 50 + %arg6 * 5] : memref<?xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 5 {
          %0 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %alloca_0[%arg4, %arg7, %arg6] : memref<2x4x5xf64>
            %2 = affine.load %arg2[%arg7 + %arg5 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %arg3[%arg5 + %arg4 * 50 + %arg6 * 5 + 25] : memref<?xf64>
        }
      }
    }
    return
  }
  func.func @mfem_integrate_grad_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x4x4xf64>
    %alloca_0 = memref.alloca() : memref<2x4x4xf64>
    %alloca_1 = memref.alloca() : memref<2x5x4xf64>
    %alloca_2 = memref.alloca() : memref<2x5x4xf64>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %arg0[%arg5 + %arg4 * 50 + %arg7 * 5] : memref<?xf64>
            %2 = affine.load %arg2[%arg6 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_2[%arg4, %arg5, %arg6] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 5 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %arg0[%arg5 + %arg4 * 50 + %arg7 * 5 + 25] : memref<?xf64>
            %2 = affine.load %arg1[%arg6 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_1[%arg4, %arg5, %arg6] : memref<2x5x4xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %alloca_2[%arg4, %arg7, %arg6] : memref<2x5x4xf64>
            %2 = affine.load %arg1[%arg5 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca_0[%arg4, %arg5, %arg6] : memref<2x4x4xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %cst) -> (f64) {
            %1 = affine.load %alloca_1[%arg4, %arg7, %arg6] : memref<2x5x4xf64>
            %2 = affine.load %arg2[%arg5 + %arg7 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg8, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca[%arg4, %arg5, %arg6] : memref<2x4x4xf64>
        }
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %alloca_0[%arg4, %arg5, %arg6] : memref<2x4x4xf64>
          %1 = affine.load %alloca[%arg4, %arg5, %arg6] : memref<2x4x4xf64>
          %2 = arith.addf %0, %1 : f64
          %3 = affine.load %arg3[%arg6 + %arg4 * 16 + %arg5 * 4] : memref<?xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg3[%arg6 + %arg4 * 16 + %arg5 * 4] : memref<?xf64>
        }
      }
    }
    return
  }
}
