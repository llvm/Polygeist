module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multilabel_margin_loss_backward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x4xi32>, %arg2: memref<?xf32>, %arg3: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.600000e+01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : f32
    %1 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %0) -> (f32) {
      affine.for %arg6 = 0 to 16 {
        affine.store %cst_1, %arg3[%arg4, %arg6] : memref<?x16xf32>
      }
      %2 = affine.for %arg6 = 0 to 4 iter_args(%arg7 = %arg5) -> (f32) {
        %3 = affine.load %arg1[%arg4, %arg6] : memref<?x4xi32>
        %4 = arith.index_cast %3 : i32 to index
        %5 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %arg7) -> (f32) {
          %6 = arith.index_cast %arg8 : index to i32
          %7 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %c0_i32) -> (i32) {
            %10 = affine.load %arg1[%arg4, %arg10] : memref<?x4xi32>
            %11 = arith.cmpi eq, %10, %6 : i32
            %12 = arith.extui %11 : i1 to i32
            %13 = arith.ori %arg11, %12 : i32
            affine.yield %13 : i32
          }
          %8 = arith.cmpi ne, %7, %c0_i32 : i32
          %9 = scf.if %8 -> (f32) {
            scf.yield %arg9 : f32
          } else {
            %10 = memref.load %arg0[%arg4, %4] : memref<?x16xf32>
            %11 = arith.subf %cst_0, %10 : f32
            %12 = affine.load %arg0[%arg4, %arg8] : memref<?x16xf32>
            %13 = arith.addf %11, %12 : f32
            %14 = arith.cmpf ogt, %13, %cst_1 : f32
            %15 = scf.if %14 -> (f32) {
              %16 = affine.load %arg2[%arg4] : memref<?xf32>
              %17 = arith.divf %16, %cst : f32
              %18 = affine.load %arg3[%arg4, %arg8] : memref<?x16xf32>
              %19 = arith.addf %18, %17 : f32
              affine.store %19, %arg3[%arg4, %arg8] : memref<?x16xf32>
              %20 = memref.load %arg3[%arg4, %4] : memref<?x16xf32>
              %21 = arith.subf %20, %17 : f32
              memref.store %21, %arg3[%arg4, %4] : memref<?x16xf32>
              scf.yield %17 : f32
            } else {
              scf.yield %arg9 : f32
            }
            scf.yield %15 : f32
          }
          affine.yield %9 : f32
        }
        affine.yield %5 : f32
      }
      affine.yield %2 : f32
    }
    return
  }
}
