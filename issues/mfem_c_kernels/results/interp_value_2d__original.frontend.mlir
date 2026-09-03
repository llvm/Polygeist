module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_interp_value_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x5xf64>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 5 {
          %0 = affine.for %arg6 = 0 to 4 iter_args(%arg7 = %cst) -> (f64) {
            %1 = affine.load %arg1[%arg6 + %arg5 * 4] : memref<?xf64>
            %2 = affine.load %arg0[%arg6 + %arg3 * 16 + %arg4 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg7, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %alloca[%arg4, %arg5] : memref<4x5xf64>
        }
      }
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 5 {
          %0 = affine.for %arg6 = 0 to 4 iter_args(%arg7 = %cst) -> (f64) {
            %1 = affine.load %alloca[%arg6, %arg4] : memref<4x5xf64>
            %2 = affine.load %arg1[%arg6 + %arg5 * 4] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %arg7, %3 : f64
            affine.yield %4 : f64
          }
          affine.store %0, %arg2[%arg5 + %arg3 * 25 + %arg4 * 5] : memref<?xf64>
        }
      }
    }
    return
  }
}
