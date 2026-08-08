#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 3 {
          affine.for %arg5 = 0 to 4 {
            %0 = arith.index_cast %arg5 : index to i32
            %1 = arith.muli %0, %c2_i32 : i32
            %2 = arith.addi %1, %c2_i32 : i32
            %3 = arith.index_cast %2 : i32 to index
            %4 = arith.index_cast %1 : i32 to index
            %5 = arith.subi %3, %4 : index
            %6:2 = affine.for %arg6 = #map(%arg3) to #map1(%arg3) iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, f32) {
              %9:2 = affine.for %arg9 = #map(%arg4) to #map1(%arg4) iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (i32, f32) {
                %10 = arith.index_cast %arg10 : i32 to index
                %11 = arith.addi %10, %5 : index
                %12 = arith.index_cast %11 : index to i32
                %13 = affine.for %arg12 = #map(%arg5) to #map1(%arg5) iter_args(%arg13 = %arg11) -> (f32) {
                  %14 = affine.load %arg0[%arg6 * 56 + %arg12 + %arg2 * 336 + %arg9 * 8] : memref<?xf32>
                  %15 = arith.addf %arg13, %14 : f32
                  affine.yield %15 : f32
                }
                affine.yield %12, %13 : i32, f32
              }
              affine.yield %9#0, %9#1 : i32, f32
            }
            %7 = arith.sitofp %6#0 : i32 to f32
            %8 = arith.divf %6#1, %7 : f32
            affine.store %8, %arg1[%arg2 * 36 + %arg5 + %arg3 * 12 + %arg4 * 4] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}
