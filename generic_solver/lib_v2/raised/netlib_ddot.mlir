module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_ddot(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.remsi %0, %c5 : index
    %2 = arith.addi %1, %c5 : index
    %3 = arith.cmpi slt, %1, %c0 : index
    %4 = arith.select %3, %2, %1 : index
    %5 = llvm.mlir.undef : f64
    %6 = arith.remsi %arg0, %c5_i32 : i32
    %7 = arith.cmpi ne, %6, %c0_i32 : i32
    %8:3 = scf.if %7 -> (f64, i1, f64) {
      %11 = affine.for %arg3 = 0 to %4 iter_args(%arg4 = %cst) -> (f64) {
        %15 = affine.load %arg1[%arg3] : memref<?xf64>
        %16 = affine.load %arg2[%arg3] : memref<?xf64>
        %17 = arith.mulf %15, %16 : f64
        %18 = arith.addf %arg4, %17 : f64
        affine.yield %18 : f64
      }
      %12 = arith.cmpi slt, %arg0, %c5_i32 : i32
      %13 = arith.cmpi sge, %arg0, %c5_i32 : i32
      %14 = arith.select %12, %11, %5 : f64
      scf.yield %11, %13, %14 : f64, i1, f64
    } else {
      scf.yield %cst, %true, %5 : f64, i1, f64
    }
    %9 = scf.if %8#1 -> (f64) {
      %11 = affine.for %arg3 = %4 to %0 step 5 iter_args(%arg4 = %8#0) -> (f64) {
        %12 = affine.load %arg1[%arg3] : memref<?xf64>
        %13 = affine.load %arg2[%arg3] : memref<?xf64>
        %14 = arith.mulf %12, %13 : f64
        %15 = arith.addf %arg4, %14 : f64
        %16 = affine.load %arg1[%arg3 + 1] : memref<?xf64>
        %17 = affine.load %arg2[%arg3 + 1] : memref<?xf64>
        %18 = arith.mulf %16, %17 : f64
        %19 = arith.addf %15, %18 : f64
        %20 = affine.load %arg1[%arg3 + 2] : memref<?xf64>
        %21 = affine.load %arg2[%arg3 + 2] : memref<?xf64>
        %22 = arith.mulf %20, %21 : f64
        %23 = arith.addf %19, %22 : f64
        %24 = affine.load %arg1[%arg3 + 3] : memref<?xf64>
        %25 = affine.load %arg2[%arg3 + 3] : memref<?xf64>
        %26 = arith.mulf %24, %25 : f64
        %27 = arith.addf %23, %26 : f64
        %28 = affine.load %arg1[%arg3 + 4] : memref<?xf64>
        %29 = affine.load %arg2[%arg3 + 4] : memref<?xf64>
        %30 = arith.mulf %28, %29 : f64
        %31 = arith.addf %27, %30 : f64
        affine.yield %31 : f64
      }
      scf.yield %11 : f64
    } else {
      scf.yield %8#0 : f64
    }
    %10 = arith.select %8#1, %9, %8#2 : f64
    return %10 : f64
  }
}
