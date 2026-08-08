module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_weight_norm_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x32xf32>, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg6 = 0 to 8 {
      %0 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %cst) -> (f32) {
        %3 = affine.load %arg0[%arg6, %arg7] : memref<?x32xf32>
        %4 = affine.load %arg1[%arg6, %arg7] : memref<?x32xf32>
        %5 = arith.mulf %3, %4 : f32
        %6 = arith.addf %arg8, %5 : f32
        affine.yield %6 : f32
      }
      %1 = affine.load %arg3[%arg6] : memref<?xf32>
      %2 = arith.divf %0, %1 : f32
      affine.store %2, %arg5[%arg6] : memref<?xf32>
      affine.for %arg7 = 0 to 32 {
        %3 = affine.load %arg2[%arg6] : memref<?xf32>
        %4 = affine.load %arg3[%arg6] : memref<?xf32>
        %5 = arith.divf %3, %4 : f32
        %6 = affine.load %arg0[%arg6, %arg7] : memref<?x32xf32>
        %7 = affine.load %arg1[%arg6, %arg7] : memref<?x32xf32>
        %8 = arith.mulf %7, %0 : f32
        %9 = arith.mulf %4, %4 : f32
        %10 = arith.divf %8, %9 : f32
        %11 = arith.subf %6, %10 : f32
        %12 = arith.mulf %5, %11 : f32
        affine.store %12, %arg4[%arg6, %arg7] : memref<?x32xf32>
      }
    }
    return
  }
}
