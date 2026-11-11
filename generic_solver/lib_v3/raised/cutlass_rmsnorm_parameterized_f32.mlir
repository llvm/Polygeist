module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cutlass_rmsnorm_parameterized_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = affine.for %arg5 = 0 to %0 iter_args(%arg6 = %cst) -> (f32) {
      %6 = affine.load %arg1[%arg5] : memref<?xf32>
      %7 = arith.mulf %6, %6 : f32
      %8 = arith.addf %arg6, %7 : f32
      affine.yield %8 : f32
    }
    %2 = arith.sitofp %arg0 : i32 to f32
    %3 = arith.divf %1, %2 : f32
    %4 = arith.addf %3, %arg4 : f32
    %5 = math.sqrt %4 : f32
    affine.for %arg5 = 0 to %0 {
      %6 = affine.load %arg1[%arg5] : memref<?xf32>
      %7 = affine.load %arg2[%arg5] : memref<?xf32>
      %8 = arith.divf %6, %5 : f32
      %9 = arith.mulf %8, %7 : f32
      affine.store %9, %arg3[%arg5] : memref<?xf32>
    }
    return
  }
}
