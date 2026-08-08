module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_binary_cross_entropy(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.560000e+02 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    affine.store %cst_1, %arg2[0] : memref<?xf32>
    affine.for %arg3 = 0 to 256 {
      %2 = affine.load %arg1[%arg3] : memref<?xf32>
      %3 = affine.load %arg0[%arg3] : memref<?xf32>
      %4 = func.call @logf(%3) : (f32) -> f32
      %5 = arith.mulf %2, %4 : f32
      %6 = affine.load %arg1[%arg3] : memref<?xf32>
      %7 = arith.subf %cst_0, %6 : f32
      %8 = affine.load %arg0[%arg3] : memref<?xf32>
      %9 = arith.subf %cst_0, %8 : f32
      %10 = func.call @logf(%9) : (f32) -> f32
      %11 = arith.mulf %7, %10 : f32
      %12 = arith.addf %5, %11 : f32
      %13 = affine.load %arg2[0] : memref<?xf32>
      %14 = arith.subf %13, %12 : f32
      affine.store %14, %arg2[0] : memref<?xf32>
    }
    %0 = affine.load %arg2[0] : memref<?xf32>
    %1 = arith.divf %0, %cst : f32
    affine.store %1, %arg2[0] : memref<?xf32>
    return
  }
  func.func private @logf(f32) -> f32
}
