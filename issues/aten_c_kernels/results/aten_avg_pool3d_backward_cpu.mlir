#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 672 {
      affine.store %cst_0, %arg1[%arg2] : memref<?xf32>
    }
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 3 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
              affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
                affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
                  %0 = affine.load %arg0[%arg2 * 36 + %arg5 + %arg3 * 12 + %arg4 * 4] : memref<?xf32>
                  %1 = arith.divf %0, %cst : f32
                  %2 = affine.load %arg1[%arg6 * 56 + %arg8 + %arg2 * 336 + %arg7 * 8] : memref<?xf32>
                  %3 = arith.addf %2, %1 : f32
                  affine.store %3, %arg1[%arg6 * 56 + %arg8 + %arg2 * 336 + %arg7 * 8] : memref<?xf32>
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
