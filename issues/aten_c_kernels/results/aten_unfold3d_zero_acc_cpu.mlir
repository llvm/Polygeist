#set = affine_set<(d0, d1, d2, d3, d4, d5) : (-d0 - d1 + 10 >= 0, d0 + d1 - 1 >= 0, -d2 - d3 + 9 >= 0, d2 + d3 - 1 >= 0, d4 + d5 - 1 >= 0, -d4 - d5 + 8 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unfold3d_zero_acc_cpu(%arg0: memref<?x3x3x3x8x9x10xf32>, %arg1: memref<?x8x9x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "polygeist.memref2pointer"(%arg1) : (memref<?x8x9x10xf32>) -> !llvm.ptr
    affine.for %arg2 = 0 to 1440 {
      %1 = arith.index_cast %arg2 : index to i32
      %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %2 : f32, !llvm.ptr
    }
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 10 {
            affine.for %arg6 = 0 to 3 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  affine.if #set(%arg8, %arg5, %arg7, %arg4, %arg3, %arg6) {
                    %1 = affine.load %arg0[%arg2, %arg6, %arg7, %arg8, %arg3, %arg4, %arg5] : memref<?x3x3x3x8x9x10xf32>
                    %2 = affine.load %arg1[%arg2, %arg3 + %arg6 - 1, %arg4 + %arg7 - 1, %arg5 + %arg8 - 1] : memref<?x8x9x10xf32>
                    %3 = arith.addf %2, %1 : f32
                    affine.store %3, %arg1[%arg2, %arg3 + %arg6 - 1, %arg4 + %arg7 - 1, %arg5 + %arg8 - 1] : memref<?x8x9x10xf32>
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
