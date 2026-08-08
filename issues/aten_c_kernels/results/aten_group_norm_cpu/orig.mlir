module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_group_norm_cpu(%arg0: memref<?x4x2x16xf32>, %arg1: memref<?x2xf32>, %arg2: memref<?x2xf32>, %arg3: f32, %arg4: memref<?x4x2x16xf32>, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    affine.for %arg7 = 0 to 4 {
      affine.for %arg8 = 0 to 4 {
        %0 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_1) -> (f32) {
          %7 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %arg10) -> (f32) {
            %8 = affine.load %arg0[%arg7, %arg8, %arg9, %arg11] : memref<?x4x2x16xf32>
            %9 = arith.addf %arg12, %8 : f32
            affine.yield %9 : f32
          }
          affine.yield %7 : f32
        }
        %1 = arith.divf %0, %cst : f32
        affine.store %1, %arg5[%arg7, %arg8] : memref<?x4xf32>
        %2 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_1) -> (f32) {
          %7 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %arg10) -> (f32) {
            %8 = affine.load %arg0[%arg7, %arg8, %arg9, %arg11] : memref<?x4x2x16xf32>
            %9 = arith.subf %8, %1 : f32
            %10 = arith.mulf %9, %9 : f32
            %11 = arith.addf %arg12, %10 : f32
            affine.yield %11 : f32
          }
          affine.yield %7 : f32
        }
        %3 = arith.divf %2, %cst : f32
        %4 = arith.addf %3, %arg3 : f32
        %5 = math.sqrt %4 : f32
        %6 = arith.divf %cst_0, %5 : f32
        affine.store %6, %arg6[%arg7, %arg8] : memref<?x4xf32>
        affine.for %arg9 = 0 to 2 {
          affine.for %arg10 = 0 to 16 {
            %7 = affine.load %arg0[%arg7, %arg8, %arg9, %arg10] : memref<?x4x2x16xf32>
            %8 = affine.load %arg5[%arg7, %arg8] : memref<?x4xf32>
            %9 = arith.subf %7, %8 : f32
            %10 = affine.load %arg6[%arg7, %arg8] : memref<?x4xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = affine.load %arg1[%arg8, %arg9] : memref<?x2xf32>
            %13 = arith.mulf %11, %12 : f32
            %14 = affine.load %arg2[%arg8, %arg9] : memref<?x2xf32>
            %15 = arith.addf %13, %14 : f32
            affine.store %15, %arg4[%arg7, %arg8, %arg9, %arg10] : memref<?x4x2x16xf32>
          }
        }
      }
    }
    return
  }
}
