module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_group_norm_backward_cpu(%arg0: memref<?x4x2x16xf32>, %arg1: memref<?x4x2x16xf32>, %arg2: memref<?x4xf32>, %arg3: memref<?x4xf32>, %arg4: memref<?x2xf32>, %arg5: memref<?x4x2x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg6 = 0 to 4 {
      affine.for %arg7 = 0 to 4 {
        %0 = affine.load %arg2[%arg6, %arg7] : memref<?x4xf32>
        %1:2 = affine.for %arg8 = 0 to 2 iter_args(%arg9 = %cst_0, %arg10 = %cst_0) -> (f32, f32) {
          %2 = affine.load %arg4[%arg7, %arg8] : memref<?x2xf32>
          %3:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (f32, f32) {
            %4 = affine.load %arg0[%arg6, %arg7, %arg8, %arg11] : memref<?x4x2x16xf32>
            %5 = arith.mulf %4, %2 : f32
            %6 = arith.addf %arg13, %5 : f32
            %7 = affine.load %arg1[%arg6, %arg7, %arg8, %arg11] : memref<?x4x2x16xf32>
            %8 = arith.subf %7, %0 : f32
            %9 = arith.mulf %5, %8 : f32
            %10 = arith.addf %arg12, %9 : f32
            affine.yield %10, %6 : f32, f32
          }
          affine.yield %3#0, %3#1 : f32, f32
        }
        affine.for %arg8 = 0 to 2 {
          affine.for %arg9 = 0 to 16 {
            %2 = affine.load %arg0[%arg6, %arg7, %arg8, %arg9] : memref<?x4x2x16xf32>
            %3 = affine.load %arg4[%arg7, %arg8] : memref<?x2xf32>
            %4 = arith.mulf %2, %3 : f32
            %5 = affine.load %arg3[%arg6, %arg7] : memref<?x4xf32>
            %6 = arith.divf %5, %cst : f32
            %7 = arith.mulf %4, %cst : f32
            %8 = arith.subf %7, %1#1 : f32
            %9 = affine.load %arg1[%arg6, %arg7, %arg8, %arg9] : memref<?x4x2x16xf32>
            %10 = affine.load %arg2[%arg6, %arg7] : memref<?x4xf32>
            %11 = arith.subf %9, %10 : f32
            %12 = arith.mulf %11, %5 : f32
            %13 = arith.mulf %12, %5 : f32
            %14 = arith.mulf %13, %1#0 : f32
            %15 = arith.subf %8, %14 : f32
            %16 = arith.mulf %6, %15 : f32
            affine.store %16, %arg5[%arg6, %arg7, %arg8, %arg9] : memref<?x4x2x16xf32>
          }
        }
      }
    }
    return
  }
}
