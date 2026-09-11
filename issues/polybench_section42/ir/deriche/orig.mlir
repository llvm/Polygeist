module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant -2.000000e+00 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = llvm.mlir.undef : f64
    %alloca = memref.alloca() : memref<f64>
    affine.store %1, %alloca[] : memref<f64>
    %alloca_3 = memref.alloca() : memref<f64>
    affine.store %1, %alloca_3[] : memref<f64>
    %alloca_4 = memref.alloca() : memref<f64>
    affine.store %1, %alloca_4[] : memref<f64>
    %2 = arith.negf %arg2 : f64
    %3 = math.exp %2 : f64
    %4 = arith.subf %cst, %3 : f64
    %5 = arith.mulf %4, %4 : f64
    %6 = arith.mulf %arg2, %cst_0 : f64
    %7 = arith.mulf %6, %3 : f64
    %8 = arith.addf %7, %cst : f64
    %9 = math.exp %6 : f64
    %10 = arith.subf %8, %9 : f64
    %11 = arith.divf %5, %10 : f64
    %12 = arith.mulf %11, %3 : f64
    %13 = arith.subf %arg2, %cst : f64
    %14 = arith.mulf %12, %13 : f64
    %15 = math.powf %cst_0, %2 : f64
    %16 = arith.mulf %arg2, %cst_1 : f64
    %17 = math.exp %16 : f64
    %18 = arith.negf %17 : f64
    %19 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %19 {
      affine.store %cst_2, %alloca_3[] : memref<f64>
      affine.store %cst_2, %alloca[] : memref<f64>
      affine.store %cst_2, %alloca_4[] : memref<f64>
      affine.for %arg8 = 0 to %0 {
        %20 = affine.load %arg3[%arg7, %arg8] : memref<?x?xf64>
        %21 = arith.mulf %11, %20 : f64
        %22 = affine.load %alloca_4[] : memref<f64>
        %23 = arith.mulf %14, %22 : f64
        %24 = arith.addf %21, %23 : f64
        %25 = affine.load %alloca_3[] : memref<f64>
        %26 = arith.mulf %15, %25 : f64
        %27 = arith.addf %24, %26 : f64
        %28 = affine.load %alloca[] : memref<f64>
        %29 = arith.mulf %18, %28 : f64
        %30 = arith.addf %27, %29 : f64
        affine.store %30, %arg5[%arg7, %arg8] : memref<?x?xf64>
        %31 = affine.load %arg3[%arg7, %arg8] : memref<?x?xf64>
        affine.store %31, %alloca_4[] : memref<f64>
        affine.store %25, %alloca[] : memref<f64>
        %32 = affine.load %arg5[%arg7, %arg8] : memref<?x?xf64>
        affine.store %32, %alloca_3[] : memref<f64>
      }
    }
    return
  }
}
