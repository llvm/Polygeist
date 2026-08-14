module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_std_var_cpu(%arg0: memref<?x64xf32>, %arg1: i32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 6.400000e+01 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = arith.subi %c64_i32, %arg1 : i32
    %1 = arith.sitofp %0 : i32 to f32
    affine.for %arg3 = 0 to 32 {
      %2 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_0) -> (f32) {
        %7 = affine.load %arg0[%arg3, %arg4] : memref<?x64xf32>
        %8 = arith.addf %arg5, %7 : f32
        affine.yield %8 : f32
      }
      %3 = arith.divf %2, %cst : f32
      %4 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_0) -> (f32) {
        %7 = affine.load %arg0[%arg3, %arg4] : memref<?x64xf32>
        %8 = arith.subf %7, %3 : f32
        %9 = arith.mulf %8, %8 : f32
        %10 = arith.addf %arg5, %9 : f32
        affine.yield %10 : f32
      }
      %5 = arith.divf %4, %1 : f32
      %6 = math.sqrt %5 : f32
      affine.store %6, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}
