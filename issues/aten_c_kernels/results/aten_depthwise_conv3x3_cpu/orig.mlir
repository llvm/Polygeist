#set = affine_set<(d0, d1, d2, d3) : (-d0 - d1 + 16 >= 0, d0 + d1 - 1 >= 0, d2 + d3 - 1 >= 0, -d2 - d3 + 16 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_depthwise_conv3x3_cpu(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x8x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg4 = 0 to 8 {
      affine.for %arg5 = 0 to 16 {
        affine.for %arg6 = 0 to 16 {
          %0 = affine.load %arg2[%arg4] : memref<?xf32>
          %1 = affine.for %arg7 = 0 to 3 iter_args(%arg8 = %0) -> (f32) {
            %2 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %arg8) -> (f32) {
              %3 = affine.if #set(%arg9, %arg6, %arg5, %arg7) -> f32 {
                %4 = affine.load %arg0[0, %arg4, %arg5 + %arg7 - 1, %arg6 + %arg9 - 1] : memref<?x8x16x16xf32>
                %5 = affine.load %arg1[%arg4, %arg7, %arg9] : memref<?x3x3xf32>
                %6 = arith.mulf %4, %5 : f32
                %7 = arith.addf %arg10, %6 : f32
                affine.yield %7 : f32
              } else {
                affine.yield %arg10 : f32
              }
              affine.yield %3 : f32
            }
            affine.yield %2 : f32
          }
          affine.store %1, %arg3[0, %arg4, %arg5, %arg6] : memref<?x8x16x16xf32>
        }
      }
    }
    return
  }
}
