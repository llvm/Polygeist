#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x30xf64>, %arg5: memref<?x20xf64>, %arg6: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %alloca = memref.alloca() : memref<f64>
    %1 = llvm.mlir.undef : f64
    affine.store %1, %alloca[] : memref<f64>
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      affine.for %arg8 = 0 to %0 {
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg9 = 0 to #map(%arg7) {
          %13 = affine.load %arg6[%arg7, %arg8] : memref<?x30xf64>
          %14 = arith.mulf %arg2, %13 : f64
          %15 = affine.load %arg5[%arg7, %arg9] : memref<?x20xf64>
          %16 = arith.mulf %14, %15 : f64
          %17 = affine.load %arg4[%arg9, %arg8] : memref<?x30xf64>
          %18 = arith.addf %17, %16 : f64
          affine.store %18, %arg4[%arg9, %arg8] : memref<?x30xf64>
          %19 = affine.load %arg6[%arg9, %arg8] : memref<?x30xf64>
          %20 = affine.load %arg5[%arg7, %arg9] : memref<?x20xf64>
          %21 = arith.mulf %19, %20 : f64
          %22 = affine.load %alloca[] : memref<f64>
          %23 = arith.addf %22, %21 : f64
          affine.store %23, %alloca[] : memref<f64>
        }
        %3 = affine.load %arg4[%arg7, %arg8] : memref<?x30xf64>
        %4 = arith.mulf %arg3, %3 : f64
        %5 = affine.load %arg6[%arg7, %arg8] : memref<?x30xf64>
        %6 = arith.mulf %arg2, %5 : f64
        %7 = affine.load %arg5[%arg7, %arg7] : memref<?x20xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %4, %8 : f64
        %10 = affine.load %alloca[] : memref<f64>
        %11 = arith.mulf %arg2, %10 : f64
        %12 = arith.addf %9, %11 : f64
        affine.store %12, %arg4[%arg7, %arg8] : memref<?x30xf64>
      }
    }
    return
  }
}
