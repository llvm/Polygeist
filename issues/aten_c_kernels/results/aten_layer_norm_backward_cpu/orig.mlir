module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_layer_norm_backward_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x64xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg8 = 0 to 64 {
      affine.store %cst_0, %arg6[%arg8] : memref<?xf32>
      affine.store %cst_0, %arg7[%arg8] : memref<?xf32>
    }
    affine.for %arg8 = 0 to 16 {
      %0:2 = affine.for %arg9 = 0 to 64 iter_args(%arg10 = %cst_0, %arg11 = %cst_0) -> (f32, f32) {
        %1 = affine.load %arg0[%arg8, %arg9] : memref<?x64xf32>
        %2 = affine.load %arg4[%arg9] : memref<?xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %arg11, %3 : f32
        %5 = affine.load %arg1[%arg8, %arg9] : memref<?x64xf32>
        %6 = affine.load %arg2[%arg8] : memref<?xf32>
        %7 = arith.subf %5, %6 : f32
        %8 = arith.mulf %3, %7 : f32
        %9 = arith.addf %arg10, %8 : f32
        %10 = arith.mulf %1, %7 : f32
        %11 = affine.load %arg3[%arg8] : memref<?xf32>
        %12 = arith.mulf %10, %11 : f32
        %13 = affine.load %arg6[%arg9] : memref<?xf32>
        %14 = arith.addf %13, %12 : f32
        affine.store %14, %arg6[%arg9] : memref<?xf32>
        %15 = affine.load %arg0[%arg8, %arg9] : memref<?x64xf32>
        %16 = affine.load %arg7[%arg9] : memref<?xf32>
        %17 = arith.addf %16, %15 : f32
        affine.store %17, %arg7[%arg9] : memref<?xf32>
        affine.yield %9, %4 : f32, f32
      }
      affine.for %arg9 = 0 to 64 {
        %1 = affine.load %arg0[%arg8, %arg9] : memref<?x64xf32>
        %2 = affine.load %arg4[%arg9] : memref<?xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = affine.load %arg3[%arg8] : memref<?xf32>
        %5 = arith.divf %4, %cst : f32
        %6 = arith.mulf %3, %cst : f32
        %7 = arith.subf %6, %0#1 : f32
        %8 = affine.load %arg1[%arg8, %arg9] : memref<?x64xf32>
        %9 = affine.load %arg2[%arg8] : memref<?xf32>
        %10 = arith.subf %8, %9 : f32
        %11 = arith.mulf %10, %4 : f32
        %12 = arith.mulf %11, %4 : f32
        %13 = arith.mulf %12, %0#0 : f32
        %14 = arith.subf %7, %13 : f32
        %15 = arith.mulf %5, %14 : f32
        affine.store %15, %arg5[%arg8, %arg9] : memref<?x64xf32>
      }
    }
    return
  }
}
