module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_conv3d(%arg0: memref<?x2x6x6x6xf32>, %arg1: memref<?x2x3x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x3x4x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg4 = 0 to 3 {
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %0 = affine.load %arg2[%arg4] : memref<?xf32>
            affine.store %0, %arg3[0, %arg4, %arg5, %arg6, %arg7] : memref<?x3x4x4x4xf32>
            affine.for %arg8 = 0 to 2 {
              affine.for %arg9 = 0 to 3 {
                affine.for %arg10 = 0 to 3 {
                  affine.for %arg11 = 0 to 3 {
                    %1 = affine.load %arg0[0, %arg8, %arg5 + %arg9, %arg6 + %arg10, %arg7 + %arg11] : memref<?x2x6x6x6xf32>
                    %2 = affine.load %arg1[%arg4, %arg8, %arg9, %arg10, %arg11] : memref<?x2x3x3x3xf32>
                    %3 = arith.mulf %1, %2 : f32
                    %4 = affine.load %arg3[0, %arg4, %arg5, %arg6, %arg7] : memref<?x3x4x4x4xf32>
                    %5 = arith.addf %4, %3 : f32
                    affine.store %5, %arg3[0, %arg4, %arg5, %arg6, %arg7] : memref<?x3x4x4x4xf32>
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
