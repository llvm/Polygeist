#set = affine_set<()[s0] : (s0 - 1 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multi_margin_loss_backward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 1.600000e+01 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    affine.for %arg7 = 0 to 32 {
      %1 = affine.load %arg1[%arg7] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst) -> (f32) {
        %5 = arith.index_cast %arg8 : index to i32
        affine.store %cst, %arg6[%arg7, %arg8] : memref<?x16xf32>
        %6 = arith.cmpi ne, %5, %1 : i32
        %7 = scf.if %6 -> (f32) {
          %8 = memref.load %arg0[%arg7, %2] : memref<?x16xf32>
          %9 = arith.subf %arg3, %8 : f32
          %10 = affine.load %arg0[%arg7, %arg8] : memref<?x16xf32>
          %11 = arith.addf %9, %10 : f32
          %12 = arith.cmpf ogt, %11, %cst : f32
          %13 = scf.if %12 -> (f32) {
            %14 = affine.load %arg5[%arg7] : memref<?xf32>
            %15 = memref.load %arg2[%2] : memref<?xf32>
            %16 = arith.mulf %14, %15 : f32
            %17 = affine.if #set()[%0] -> f32 {
              affine.yield %cst_0 : f32
            } else {
              %21 = arith.mulf %11, %cst_1 : f32
              affine.yield %21 : f32
            }
            %18 = arith.mulf %16, %17 : f32
            %19 = arith.divf %18, %cst_2 : f32
            affine.store %19, %arg6[%arg7, %arg8] : memref<?x16xf32>
            %20 = arith.addf %arg9, %19 : f32
            scf.yield %20 : f32
          } else {
            scf.yield %arg9 : f32
          }
          scf.yield %13 : f32
        } else {
          scf.yield %arg9 : f32
        }
        affine.yield %7 : f32
      }
      %4 = arith.negf %3 : f32
      memref.store %4, %arg6[%arg7, %2] : memref<?x16xf32>
    }
    return
  }
}
