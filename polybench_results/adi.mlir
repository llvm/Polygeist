#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<?x20xf64>, %arg3: memref<?x20xf64>, %arg4: memref<?x20xf64>, %arg5: memref<?x20xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.sitofp %arg1 : i32 to f64
    %2 = arith.divf %cst, %1 : f64
    %3 = arith.sitofp %arg0 : i32 to f64
    %4 = arith.divf %cst, %3 : f64
    %5 = arith.mulf %4, %cst_0 : f64
    %6 = arith.mulf %2, %2 : f64
    %7 = arith.divf %5, %6 : f64
    %8 = arith.divf %4, %6 : f64
    %9 = arith.negf %7 : f64
    %10 = arith.divf %9, %cst_0 : f64
    %11 = arith.addf %7, %cst : f64
    %12 = arith.negf %8 : f64
    %13 = arith.divf %12, %cst_0 : f64
    %14 = arith.addf %8, %cst : f64
    %15 = arith.index_cast %arg0 : i32 to index
    %16 = arith.negf %10 : f64
    %17 = arith.negf %13 : f64
    %18 = arith.mulf %13, %cst_0 : f64
    %19 = arith.addf %18, %cst : f64
    %20 = arith.mulf %10, %cst_0 : f64
    %21 = arith.addf %20, %cst : f64
    affine.for %arg6 = 1 to #map()[%15] {
      affine.for %arg7 = 1 to #map1()[%0] {
        affine.store %cst, %arg3[0, %arg7] : memref<?x20xf64>
        affine.store %cst_1, %arg4[%arg7, 0] : memref<?x20xf64>
        %22 = affine.load %arg3[0, %arg7] : memref<?x20xf64>
        affine.store %22, %arg5[%arg7, 0] : memref<?x20xf64>
        affine.for %arg8 = 1 to #map1()[%0] {
          %23 = affine.load %arg4[%arg7, %arg8 - 1] : memref<?x20xf64>
          %24 = arith.mulf %10, %23 : f64
          %25 = arith.addf %24, %11 : f64
          %26 = arith.divf %16, %25 : f64
          affine.store %26, %arg4[%arg7, %arg8] : memref<?x20xf64>
          %27 = affine.load %arg2[%arg8, %arg7 - 1] : memref<?x20xf64>
          %28 = arith.mulf %17, %27 : f64
          %29 = affine.load %arg2[%arg8, %arg7] : memref<?x20xf64>
          %30 = arith.mulf %19, %29 : f64
          %31 = arith.addf %28, %30 : f64
          %32 = affine.load %arg2[%arg8, %arg7 + 1] : memref<?x20xf64>
          %33 = arith.mulf %13, %32 : f64
          %34 = arith.subf %31, %33 : f64
          %35 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x20xf64>
          %36 = arith.mulf %10, %35 : f64
          %37 = arith.subf %34, %36 : f64
          %38 = arith.divf %37, %25 : f64
          affine.store %38, %arg5[%arg7, %arg8] : memref<?x20xf64>
        }
        affine.store %cst, %arg3[symbol(%0) - 1, %arg7] : memref<?x20xf64>
        affine.for %arg8 = 1 to #map1()[%0] {
          %23 = affine.load %arg4[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x20xf64>
          %24 = affine.load %arg3[-%arg8 + symbol(%0), %arg7] : memref<?x20xf64>
          %25 = arith.mulf %23, %24 : f64
          %26 = affine.load %arg5[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x20xf64>
          %27 = arith.addf %25, %26 : f64
          affine.store %27, %arg3[-%arg8 + symbol(%0) - 1, %arg7] : memref<?x20xf64>
        }
      }
      affine.for %arg7 = 1 to #map1()[%0] {
        affine.store %cst, %arg2[%arg7, 0] : memref<?x20xf64>
        affine.store %cst_1, %arg4[%arg7, 0] : memref<?x20xf64>
        %22 = affine.load %arg2[%arg7, 0] : memref<?x20xf64>
        affine.store %22, %arg5[%arg7, 0] : memref<?x20xf64>
        affine.for %arg8 = 1 to #map1()[%0] {
          %23 = affine.load %arg4[%arg7, %arg8 - 1] : memref<?x20xf64>
          %24 = arith.mulf %13, %23 : f64
          %25 = arith.addf %24, %14 : f64
          %26 = arith.divf %17, %25 : f64
          affine.store %26, %arg4[%arg7, %arg8] : memref<?x20xf64>
          %27 = affine.load %arg3[%arg7 - 1, %arg8] : memref<?x20xf64>
          %28 = arith.mulf %16, %27 : f64
          %29 = affine.load %arg3[%arg7, %arg8] : memref<?x20xf64>
          %30 = arith.mulf %21, %29 : f64
          %31 = arith.addf %28, %30 : f64
          %32 = affine.load %arg3[%arg7 + 1, %arg8] : memref<?x20xf64>
          %33 = arith.mulf %10, %32 : f64
          %34 = arith.subf %31, %33 : f64
          %35 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x20xf64>
          %36 = arith.mulf %13, %35 : f64
          %37 = arith.subf %34, %36 : f64
          %38 = arith.divf %37, %25 : f64
          affine.store %38, %arg5[%arg7, %arg8] : memref<?x20xf64>
        }
        affine.store %cst, %arg2[%arg7, symbol(%0) - 1] : memref<?x20xf64>
        affine.for %arg8 = 1 to #map1()[%0] {
          %23 = affine.load %arg4[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x20xf64>
          %24 = affine.load %arg2[%arg7, -%arg8 + symbol(%0)] : memref<?x20xf64>
          %25 = arith.mulf %23, %24 : f64
          %26 = affine.load %arg5[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x20xf64>
          %27 = arith.addf %25, %26 : f64
          affine.store %27, %arg2[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x20xf64>
        }
      }
    }
    return
  }
}
