module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_mass_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<4x4xf64>
    %alloca_1 = memref.alloca() : memref<5xf64>
    %alloca_2 = memref.alloca() : memref<5x5xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.store %cst, %alloca_3[%arg6, %arg7, %arg8] : memref<5x5x5xf64>
          }
        }
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            affine.store %cst, %alloca_2[%arg7, %arg8] : memref<5x5xf64>
          }
        }
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 5 {
            affine.store %cst, %alloca_1[%arg8] : memref<5xf64>
          }
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg3[%arg5 * 64 + %arg8 + %arg6 * 16 + %arg7 * 4] : memref<?xf64>
            affine.for %arg9 = 0 to 5 {
              %1 = affine.load %arg0[%arg8 + %arg9 * 4] : memref<?xf64>
              %2 = arith.mulf %1, %0 : f64
              %3 = affine.load %alloca_1[%arg9] : memref<5xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_1[%arg9] : memref<5xf64>
            }
          }
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %arg0[%arg7 + %arg8 * 4] : memref<?xf64>
            affine.for %arg9 = 0 to 5 {
              %1 = affine.load %alloca_1[%arg9] : memref<5xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_2[%arg8, %arg9] : memref<5x5xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_2[%arg8, %arg9] : memref<5x5xf64>
            }
          }
        }
        affine.for %arg7 = 0 to 5 {
          %0 = affine.load %arg0[%arg6 + %arg7 * 4] : memref<?xf64>
          affine.for %arg8 = 0 to 5 {
            affine.for %arg9 = 0 to 5 {
              %1 = affine.load %alloca_2[%arg8, %arg9] : memref<5x5xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_3[%arg7, %arg8, %arg9] : memref<5x5x5xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_3[%arg7, %arg8, %arg9] : memref<5x5x5xf64>
            }
          }
        }
      }
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %arg2[%arg5 * 125 + %arg8 + %arg6 * 25 + %arg7 * 5] : memref<?xf64>
            %1 = affine.load %alloca_3[%arg6, %arg7, %arg8] : memref<5x5x5xf64>
            %2 = arith.mulf %1, %0 : f64
            affine.store %2, %alloca_3[%arg6, %arg7, %arg8] : memref<5x5x5xf64>
          }
        }
      }
      affine.for %arg6 = 0 to 5 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            affine.store %cst, %alloca_0[%arg7, %arg8] : memref<4x4xf64>
          }
        }
        affine.for %arg7 = 0 to 5 {
          affine.for %arg8 = 0 to 4 {
            affine.store %cst, %alloca[%arg8] : memref<4xf64>
          }
          affine.for %arg8 = 0 to 5 {
            %0 = affine.load %alloca_3[%arg6, %arg7, %arg8] : memref<5x5x5xf64>
            affine.for %arg9 = 0 to 4 {
              %1 = affine.load %arg1[%arg8 + %arg9 * 5] : memref<?xf64>
              %2 = arith.mulf %1, %0 : f64
              %3 = affine.load %alloca[%arg9] : memref<4xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca[%arg9] : memref<4xf64>
            }
          }
          affine.for %arg8 = 0 to 4 {
            %0 = affine.load %arg1[%arg7 + %arg8 * 5] : memref<?xf64>
            affine.for %arg9 = 0 to 4 {
              %1 = affine.load %alloca[%arg9] : memref<4xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %alloca_0[%arg8, %arg9] : memref<4x4xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %alloca_0[%arg8, %arg9] : memref<4x4xf64>
            }
          }
        }
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            affine.for %arg9 = 0 to 4 {
              %0 = affine.load %arg1[%arg6 + %arg7 * 5] : memref<?xf64>
              %1 = affine.load %alloca_0[%arg8, %arg9] : memref<4x4xf64>
              %2 = arith.mulf %0, %1 : f64
              %3 = affine.load %arg4[%arg5 * 64 + %arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %arg4[%arg5 * 64 + %arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
            }
          }
        }
      }
    }
    return
  }
}
