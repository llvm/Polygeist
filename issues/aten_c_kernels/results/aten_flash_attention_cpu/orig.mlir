module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flash_attention_cpu(%arg0: memref<?x2x16x32xf32>, %arg1: memref<?x2x16x32xf32>, %arg2: memref<?x2x16x32xf32>, %arg3: memref<?x2x16x32xf32>, %arg4: memref<?x2x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.176776692 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %alloca = memref.alloca() : memref<16xf32>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 16 {
        %0 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %cst_1) -> (f32) {
          %4 = affine.for %arg9 = 0 to 32 iter_args(%arg10 = %cst_0) -> (f32) {
            %8 = affine.load %arg0[0, %arg5, %arg6, %arg9] : memref<?x2x16x32xf32>
            %9 = affine.load %arg1[0, %arg5, %arg7, %arg9] : memref<?x2x16x32xf32>
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %arg10, %10 : f32
            affine.yield %11 : f32
          }
          %5 = arith.mulf %4, %cst : f32
          affine.store %5, %alloca[%arg7] : memref<16xf32>
          %6 = arith.cmpf ogt, %5, %arg8 : f32
          %7 = arith.select %6, %5, %arg8 : f32
          affine.yield %7 : f32
        }
        %1 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %cst_0) -> (f32) {
          %4 = affine.load %alloca[%arg7] : memref<16xf32>
          %5 = arith.subf %4, %0 : f32
          %6 = math.exp %5 : f32
          affine.store %6, %alloca[%arg7] : memref<16xf32>
          %7 = arith.addf %arg8, %6 : f32
          affine.yield %7 : f32
        }
        %2 = func.call @logf(%1) : (f32) -> f32
        %3 = arith.addf %0, %2 : f32
        affine.store %3, %arg4[0, %arg5, %arg6] : memref<?x2x16xf32>
        affine.for %arg7 = 0 to 32 {
          %4 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst_0) -> (f32) {
            %5 = affine.load %alloca[%arg8] : memref<16xf32>
            %6 = arith.divf %5, %1 : f32
            %7 = affine.load %arg2[0, %arg5, %arg8, %arg7] : memref<?x2x16x32xf32>
            %8 = arith.mulf %6, %7 : f32
            %9 = arith.addf %arg9, %8 : f32
            affine.yield %9 : f32
          }
          affine.store %4, %arg3[0, %arg5, %arg6, %arg7] : memref<?x2x16x32xf32>
        }
      }
    }
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
