module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_3d_cpu(%arg0: memref<?x2x6x7x8xf32>, %arg1: memref<?x4x5x6x3xf32>, %arg2: memref<?x2x4x5x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 6.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %cst_4 = arith.constant 1.000000e+00 : f32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 6 {
          %0 = affine.load %arg1[0, %arg3, %arg4, %arg5, 0] : memref<?x4x5x6x3xf32>
          %1 = arith.addf %0, %cst_4 : f32
          %2 = arith.mulf %1, %cst_3 : f32
          %3 = arith.mulf %2, %cst : f32
          %4 = affine.load %arg1[0, %arg3, %arg4, %arg5, 1] : memref<?x4x5x6x3xf32>
          %5 = arith.addf %4, %cst_4 : f32
          %6 = arith.mulf %5, %cst_3 : f32
          %7 = arith.mulf %6, %cst_0 : f32
          %8 = affine.load %arg1[0, %arg3, %arg4, %arg5, 2] : memref<?x4x5x6x3xf32>
          %9 = arith.addf %8, %cst_4 : f32
          %10 = arith.mulf %9, %cst_3 : f32
          %11 = arith.mulf %10, %cst_1 : f32
          %12 = arith.fptosi %3 : f32 to i32
          %13 = arith.fptosi %7 : f32 to i32
          %14 = arith.fptosi %11 : f32 to i32
          %15 = arith.sitofp %12 : i32 to f32
          %16 = arith.subf %3, %15 : f32
          %17 = arith.sitofp %13 : i32 to f32
          %18 = arith.subf %7, %17 : f32
          %19 = arith.sitofp %14 : i32 to f32
          %20 = arith.subf %11, %19 : f32
          %21 = arith.subf %cst_4, %16 : f32
          %22 = arith.subf %cst_4, %18 : f32
          %23 = arith.subf %cst_4, %20 : f32
          affine.for %arg6 = 0 to 2 {
            %24 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %cst_2) -> (f32) {
              %25 = arith.index_cast %arg7 : index to i32
              %26 = arith.addi %14, %25 : i32
              %27 = arith.cmpi sge, %26, %c0_i32 : i32
              %28 = arith.cmpi slt, %26, %c6_i32 : i32
              %29 = arith.index_cast %26 : i32 to index
              %30 = arith.cmpi ne, %25, %c0_i32 : i32
              %31 = arith.select %30, %20, %23 : f32
              %32 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %arg8) -> (f32) {
                %33 = arith.index_cast %arg9 : index to i32
                %34 = arith.addi %13, %33 : i32
                %35 = arith.cmpi sge, %34, %c0_i32 : i32
                %36 = arith.cmpi slt, %34, %c7_i32 : i32
                %37 = arith.index_cast %34 : i32 to index
                %38 = arith.cmpi ne, %33, %c0_i32 : i32
                %39 = arith.select %38, %18, %22 : f32
                %40 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %arg10) -> (f32) {
                  %41 = arith.index_cast %arg11 : index to i32
                  %42 = arith.addi %12, %41 : i32
                  %43 = arith.cmpi sge, %42, %c0_i32 : i32
                  %44 = arith.cmpi slt, %42, %c8_i32 : i32
                  %45 = arith.andi %43, %44 : i1
                  %46 = arith.andi %36, %45 : i1
                  %47 = arith.andi %35, %46 : i1
                  %48 = arith.andi %28, %47 : i1
                  %49 = arith.andi %27, %48 : i1
                  %50 = scf.if %49 -> (f32) {
                    %51 = arith.index_cast %42 : i32 to index
                    %52 = memref.load %arg0[%c0, %arg6, %29, %37, %51] : memref<?x2x6x7x8xf32>
                    %53 = arith.mulf %52, %31 : f32
                    %54 = arith.mulf %53, %39 : f32
                    %55 = arith.cmpi ne, %41, %c0_i32 : i32
                    %56 = arith.select %55, %16, %21 : f32
                    %57 = arith.mulf %54, %56 : f32
                    %58 = arith.addf %arg12, %57 : f32
                    scf.yield %58 : f32
                  } else {
                    scf.yield %arg12 : f32
                  }
                  affine.yield %50 : f32
                }
                affine.yield %40 : f32
              }
              affine.yield %32 : f32
            }
            affine.store %24, %arg2[0, %arg6, %arg3, %arg4, %arg5] : memref<?x2x4x5x6xf32>
          }
        }
      }
    }
    return
  }
}
