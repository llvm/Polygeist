module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_backward_cpu(%arg0: memref<?x8x32xf32>, %arg1: memref<?x8x32xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x8x32xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.280000e+02 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg8 = 0 to 8 {
      %0 = affine.load %arg2[%arg8] : memref<?xf32>
      %1:2 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst_0, %arg11 = %cst_0) -> (f32, f32) {
        %4:2 = affine.for %arg12 = 0 to 32 iter_args(%arg13 = %arg10, %arg14 = %arg11) -> (f32, f32) {
          %5 = affine.load %arg0[%arg9, %arg8, %arg12] : memref<?x8x32xf32>
          %6 = arith.addf %arg14, %5 : f32
          %7 = affine.load %arg1[%arg9, %arg8, %arg12] : memref<?x8x32xf32>
          %8 = arith.subf %7, %0 : f32
          %9 = arith.mulf %5, %8 : f32
          %10 = arith.addf %arg13, %9 : f32
          affine.yield %10, %6 : f32, f32
        }
        affine.yield %4#0, %4#1 : f32, f32
      }
      affine.store %1#1, %arg7[%arg8] : memref<?xf32>
      %2 = affine.load %arg3[%arg8] : memref<?xf32>
      %3 = arith.mulf %1#0, %2 : f32
      affine.store %3, %arg6[%arg8] : memref<?xf32>
      affine.for %arg9 = 0 to 4 {
        affine.for %arg10 = 0 to 32 {
          %4 = affine.load %arg4[%arg8] : memref<?xf32>
          %5 = affine.load %arg3[%arg8] : memref<?xf32>
          %6 = arith.mulf %4, %5 : f32
          %7 = arith.divf %6, %cst : f32
          %8 = affine.load %arg0[%arg9, %arg8, %arg10] : memref<?x8x32xf32>
          %9 = arith.mulf %8, %cst : f32
          %10 = arith.subf %9, %1#1 : f32
          %11 = affine.load %arg1[%arg9, %arg8, %arg10] : memref<?x8x32xf32>
          %12 = affine.load %arg2[%arg8] : memref<?xf32>
          %13 = arith.subf %11, %12 : f32
          %14 = arith.mulf %13, %5 : f32
          %15 = arith.mulf %14, %5 : f32
          %16 = arith.mulf %15, %1#0 : f32
          %17 = arith.subf %10, %16 : f32
          %18 = arith.mulf %7, %17 : f32
          affine.store %18, %arg5[%arg9, %arg8, %arg10] : memref<?x8x32xf32>
        }
      }
    }
    return
  }
}
