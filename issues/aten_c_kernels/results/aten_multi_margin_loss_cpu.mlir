#set = affine_set<()[s0] : (s0 - 1 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multi_margin_loss_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.600000e+01 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    affine.for %arg6 = 0 to 32 {
      %1 = affine.load %arg1[%arg6] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %cst) -> (f32) {
        %7 = arith.index_cast %arg7 : index to i32
        %8 = arith.cmpi ne, %7, %1 : i32
        %9 = scf.if %8 -> (f32) {
          %10 = memref.load %arg0[%arg6, %2] : memref<?x16xf32>
          %11 = arith.subf %arg3, %10 : f32
          %12 = affine.load %arg0[%arg6, %arg7] : memref<?x16xf32>
          %13 = arith.addf %11, %12 : f32
          %14 = arith.cmpf ogt, %13, %cst : f32
          %15 = scf.if %14 -> (f32) {
            %16 = affine.if #set()[%0] -> f32 {
              affine.yield %13 : f32
            } else {
              %18 = arith.mulf %13, %13 : f32
              affine.yield %18 : f32
            }
            %17 = arith.addf %arg8, %16 : f32
            scf.yield %17 : f32
          } else {
            scf.yield %arg8 : f32
          }
          scf.yield %15 : f32
        } else {
          scf.yield %arg8 : f32
        }
        affine.yield %9 : f32
      }
      %4 = memref.load %arg2[%2] : memref<?xf32>
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.divf %5, %cst_0 : f32
      affine.store %6, %arg5[%arg6] : memref<?xf32>
    }
    return
  }
}
