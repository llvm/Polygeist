module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x2xf64>
    %alloca_1 = memref.alloca() : memref<2x2xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 25 {
        %0 = affine.load %arg2[%arg6 + %arg5 * 100] : memref<?xf64>
        %1 = affine.load %arg2[%arg6 + %arg5 * 100 + 25] : memref<?xf64>
        %2 = affine.load %arg2[%arg6 + %arg5 * 100 + 50] : memref<?xf64>
        %3 = affine.load %arg2[%arg6 + %arg5 * 100 + 75] : memref<?xf64>
        %4 = arith.mulf %0, %3 : f64
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.subf %4, %5 : f64
        %7 = arith.divf %3, %6 : f64
        affine.store %7, %alloca_1[0, 0] : memref<2x2xf64>
        %8 = arith.negf %1 : f64
        %9 = arith.divf %8, %6 : f64
        affine.store %9, %alloca_1[0, 1] : memref<2x2xf64>
        %10 = arith.negf %2 : f64
        %11 = arith.divf %10, %6 : f64
        affine.store %11, %alloca_1[1, 0] : memref<2x2xf64>
        %12 = arith.divf %0, %6 : f64
        affine.store %12, %alloca_1[1, 1] : memref<2x2xf64>
        %13 = affine.load %arg4[%arg6 + %arg5 * 100] : memref<?xf64>
        %14 = affine.load %arg4[%arg6 + %arg5 * 100 + 75] : memref<?xf64>
        %15 = arith.addf %13, %14 : f64
        %16 = affine.load %arg3[%arg6] : memref<?xf64>
        %17 = arith.mulf %16, %6 : f64
        %18 = affine.load %arg0[%arg6 + %arg5 * 25] : memref<?xf64>
        %19 = affine.load %arg1[%arg6 + %arg5 * 25] : memref<?xf64>
        %20 = arith.mulf %19, %cst : f64
        affine.for %arg7 = 0 to 2 {
          affine.for %arg8 = 0 to 2 {
            %21 = arith.index_cast %arg8 : index to i32
            %22 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_0) -> (f64) {
              %29 = arith.index_cast %arg9 : index to i32
              %30 = arith.cmpi eq, %29, %21 : i32
              %31 = arith.extui %30 : i1 to i32
              %32 = arith.sitofp %31 : i32 to f64
              %33 = affine.load %alloca_1[%arg7, %arg9] : memref<2x2xf64>
              %34 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %arg10) -> (f64) {
                %35 = arith.index_cast %arg11 : index to i32
                %36 = affine.load %alloca_1[%arg7, %arg11] : memref<2x2xf64>
                %37 = arith.mulf %32, %36 : f64
                %38 = arith.cmpi eq, %35, %21 : i32
                %39 = arith.extui %38 : i1 to i32
                %40 = arith.sitofp %39 : i32 to f64
                %41 = arith.mulf %40, %33 : f64
                %42 = arith.addf %37, %41 : f64
                %43 = affine.load %arg4[%arg5 * 100 + %arg6 + %arg9 * 50 + %arg11 * 25] : memref<?xf64>
                %44 = affine.load %arg4[%arg5 * 100 + %arg6 + %arg11 * 50 + %arg9 * 25] : memref<?xf64>
                %45 = arith.addf %43, %44 : f64
                %46 = arith.mulf %42, %45 : f64
                %47 = arith.addf %arg12, %46 : f64
                affine.yield %47 : f64
              }
              affine.yield %34 : f64
            }
            %23 = affine.load %alloca_1[%arg7, %arg8] : memref<2x2xf64>
            %24 = arith.mulf %18, %23 : f64
            %25 = arith.mulf %24, %15 : f64
            %26 = arith.mulf %20, %22 : f64
            %27 = arith.addf %25, %26 : f64
            %28 = arith.mulf %17, %27 : f64
            affine.store %28, %alloca[%arg7, %arg8] : memref<2x2xf64>
          }
        }
        affine.for %arg7 = 0 to 2 {
          affine.for %arg8 = 0 to 2 {
            %21 = affine.load %alloca[%arg7, %arg8] : memref<2x2xf64>
            affine.store %21, %arg4[%arg5 * 100 + %arg6 + %arg8 * 50 + %arg7 * 25] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
