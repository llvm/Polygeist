module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_slow_conv3d_forward_cpu(%arg0: memref<?x8x9x10xf32>, %arg1: memref<?x2x3x3x3xf32>, %arg2: memref<?x6x7x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 6 {
        affine.for %arg5 = 0 to 7 {
          affine.for %arg6 = 0 to 8 {
            %0 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %cst) -> (f32) {
              %1 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %arg8) -> (f32) {
                %2 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %arg10) -> (f32) {
                  %3 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %arg12) -> (f32) {
                    %4 = affine.load %arg0[%arg7, %arg4 + %arg9, %arg5 + %arg11, %arg6 + %arg13] : memref<?x8x9x10xf32>
                    %5 = affine.load %arg1[%arg3, %arg7, %arg9, %arg11, %arg13] : memref<?x2x3x3x3xf32>
                    %6 = arith.mulf %4, %5 : f32
                    %7 = arith.addf %arg14, %6 : f32
                    affine.yield %7 : f32
                  }
                  affine.yield %3 : f32
                }
                affine.yield %2 : f32
              }
              affine.yield %1 : f32
            }
            affine.store %0, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x6x7x8xf32>
          }
        }
      }
    }
    return
  }
}
