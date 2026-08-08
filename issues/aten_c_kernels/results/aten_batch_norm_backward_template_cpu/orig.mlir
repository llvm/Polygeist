module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_backward_template_cpu(%arg0: memref<?x16x16x16xf32>, %arg1: memref<?x16x16x16xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x16x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.048000e+03 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg5 = 0 to 16 {
      %0:2 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %cst_0, %arg8 = %cst_0) -> (f32, f32) {
        %2 = affine.load %arg2[%arg5] : memref<?xf32>
        %3:2 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (f32, f32) {
          %4:2 = affine.for %arg12 = 0 to 16 iter_args(%arg13 = %arg10, %arg14 = %arg11) -> (f32, f32) {
            %5 = affine.load %arg0[%arg6, %arg5, %arg9, %arg12] : memref<?x16x16x16xf32>
            %6 = arith.addf %arg14, %5 : f32
            %7 = affine.load %arg1[%arg6, %arg5, %arg9, %arg12] : memref<?x16x16x16xf32>
            %8 = arith.subf %7, %2 : f32
            %9 = arith.mulf %5, %8 : f32
            %10 = arith.addf %arg13, %9 : f32
            affine.yield %10, %6 : f32, f32
          }
          affine.yield %4#0, %4#1 : f32, f32
        }
        affine.yield %3#0, %3#1 : f32, f32
      }
      %1 = arith.divf %0#1, %cst : f32
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 16 {
          affine.for %arg8 = 0 to 16 {
            %2 = affine.load %arg3[%arg5] : memref<?xf32>
            %3 = affine.load %arg0[%arg6, %arg5, %arg7, %arg8] : memref<?x16x16x16xf32>
            %4 = arith.subf %3, %1 : f32
            %5 = affine.load %arg1[%arg6, %arg5, %arg7, %arg8] : memref<?x16x16x16xf32>
            %6 = affine.load %arg2[%arg5] : memref<?xf32>
            %7 = arith.subf %5, %6 : f32
            %8 = arith.mulf %7, %2 : f32
            %9 = arith.mulf %8, %2 : f32
            %10 = arith.mulf %9, %0#0 : f32
            %11 = arith.divf %10, %cst : f32
            %12 = arith.subf %4, %11 : f32
            %13 = arith.mulf %2, %12 : f32
            affine.store %13, %arg4[%arg6, %arg5, %arg7, %arg8] : memref<?x16x16x16xf32>
          }
        }
      }
    }
    return
  }
}
