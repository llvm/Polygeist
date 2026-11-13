module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cdist_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?x12xf32>, %arg3: memref<?x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 32 {
        affine.store %cst, %arg3[%arg4, %arg5] : memref<?x32xf32>
      }
    }
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 12 {
        %0 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %cst) -> (f32) {
          %4 = affine.load %arg0[%arg4, %arg6] : memref<?x32xf32>
          %5 = affine.load %arg1[%arg5, %arg6] : memref<?x32xf32>
          %6 = arith.subf %4, %5 : f32
          %7 = arith.mulf %6, %6 : f32
          %8 = arith.addf %arg7, %7 : f32
          affine.yield %8 : f32
        }
        %1 = math.sqrt %0 : f32
        %2 = arith.cmpf oeq, %1, %cst : f32
        %3 = scf.if %2 -> (f32) {
          scf.yield %cst : f32
        } else {
          %4 = affine.load %arg2[%arg4, %arg5] : memref<?x12xf32>
          %5 = arith.divf %4, %1 : f32
          scf.yield %5 : f32
        }
        affine.for %arg6 = 0 to 32 {
          %4 = affine.load %arg0[%arg4, %arg6] : memref<?x32xf32>
          %5 = affine.load %arg1[%arg5, %arg6] : memref<?x32xf32>
          %6 = arith.subf %4, %5 : f32
          %7 = arith.mulf %3, %6 : f32
          %8 = affine.load %arg3[%arg4, %arg6] : memref<?x32xf32>
          %9 = arith.addf %8, %7 : f32
          affine.store %9, %arg3[%arg4, %arg6] : memref<?x32xf32>
        }
      }
    }
    return
  }
}
