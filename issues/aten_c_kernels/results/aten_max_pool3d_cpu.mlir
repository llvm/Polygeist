#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_max_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 3 {
          affine.for %arg6 = 0 to 4 {
            %0:2 = affine.for %arg7 = #map(%arg4) to #map1(%arg4) iter_args(%arg8 = %c0_i32, %arg9 = %cst) -> (i32, f32) {
              %1 = arith.index_cast %arg7 : index to i32
              %2 = arith.muli %1, %c7_i32 : i32
              %3:2 = affine.for %arg10 = #map(%arg5) to #map1(%arg5) iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (i32, f32) {
                %4 = arith.index_cast %arg10 : index to i32
                %5 = arith.addi %2, %4 : i32
                %6 = arith.muli %5, %c8_i32 : i32
                %7:2 = affine.for %arg13 = #map(%arg6) to #map1(%arg6) iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (i32, f32) {
                  %8 = arith.index_cast %arg13 : index to i32
                  %9 = affine.load %arg0[%arg7 * 56 + %arg13 + %arg3 * 336 + %arg10 * 8] : memref<?xf32>
                  %10 = arith.cmpf ogt, %9, %arg15 : f32
                  %11 = arith.select %10, %9, %arg15 : f32
                  %12 = scf.if %10 -> (i32) {
                    %13 = arith.addi %6, %8 : i32
                    scf.yield %13 : i32
                  } else {
                    scf.yield %arg14 : i32
                  }
                  affine.yield %12, %11 : i32, f32
                }
                affine.yield %7#0, %7#1 : i32, f32
              }
              affine.yield %3#0, %3#1 : i32, f32
            }
            affine.store %0#1, %arg1[%arg3 * 36 + %arg6 + %arg4 * 12 + %arg5 * 4] : memref<?xf32>
            affine.store %0#0, %arg2[%arg3 * 36 + %arg6 + %arg4 * 12 + %arg5 * 4] : memref<?xi32>
          }
        }
      }
    }
    return
  }
}
