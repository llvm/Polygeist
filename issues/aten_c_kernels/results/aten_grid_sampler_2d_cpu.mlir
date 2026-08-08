module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_cpu(%arg0: memref<?x3x8x8xf32>, %arg1: memref<?x6x6x2xf32>, %arg2: memref<?x3x6x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 7.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 6 {
      affine.for %arg4 = 0 to 6 {
        %0 = affine.load %arg1[0, %arg3, %arg4, 0] : memref<?x6x6x2xf32>
        %1 = arith.addf %0, %cst_2 : f32
        %2 = arith.mulf %1, %cst_1 : f32
        %3 = arith.mulf %2, %cst : f32
        %4 = affine.load %arg1[0, %arg3, %arg4, 1] : memref<?x6x6x2xf32>
        %5 = arith.addf %4, %cst_2 : f32
        %6 = arith.mulf %5, %cst_1 : f32
        %7 = arith.mulf %6, %cst : f32
        %8 = arith.fptosi %3 : f32 to i32
        %9 = arith.fptosi %7 : f32 to i32
        %10 = arith.addi %8, %c1_i32 : i32
        %11 = arith.addi %9, %c1_i32 : i32
        %12 = arith.sitofp %8 : i32 to f32
        %13 = arith.subf %3, %12 : f32
        %14 = arith.sitofp %9 : i32 to f32
        %15 = arith.subf %7, %14 : f32
        %16 = arith.cmpi sge, %8, %c0_i32 : i32
        %17 = arith.cmpi slt, %8, %c8_i32 : i32
        %18 = arith.cmpi sge, %9, %c0_i32 : i32
        %19 = arith.cmpi slt, %9, %c8_i32 : i32
        %20 = arith.andi %18, %19 : i1
        %21 = arith.andi %17, %20 : i1
        %22 = arith.andi %16, %21 : i1
        %23 = arith.cmpi sge, %10, %c0_i32 : i32
        %24 = arith.cmpi slt, %10, %c8_i32 : i32
        %25 = arith.andi %24, %20 : i1
        %26 = arith.andi %23, %25 : i1
        %27 = arith.cmpi sge, %11, %c0_i32 : i32
        %28 = arith.cmpi slt, %11, %c8_i32 : i32
        %29 = arith.andi %27, %28 : i1
        %30 = arith.andi %17, %29 : i1
        %31 = arith.andi %16, %30 : i1
        %32 = arith.andi %24, %29 : i1
        %33 = arith.andi %23, %32 : i1
        %34 = arith.subf %cst_2, %13 : f32
        %35 = arith.subf %cst_2, %15 : f32
        %36 = arith.mulf %34, %35 : f32
        %37 = arith.index_cast %9 : i32 to index
        %38 = arith.index_cast %8 : i32 to index
        %39 = arith.mulf %13, %35 : f32
        %40 = arith.index_cast %10 : i32 to index
        %41 = arith.mulf %34, %15 : f32
        %42 = arith.index_cast %11 : i32 to index
        %43 = arith.mulf %13, %15 : f32
        affine.for %arg5 = 0 to 3 {
          %44 = scf.if %22 -> (f32) {
            %48 = memref.load %arg0[%c0, %arg5, %37, %38] : memref<?x3x8x8xf32>
            %49 = arith.mulf %36, %48 : f32
            %50 = arith.addf %49, %cst_0 : f32
            scf.yield %50 : f32
          } else {
            scf.yield %cst_0 : f32
          }
          %45 = scf.if %26 -> (f32) {
            %48 = memref.load %arg0[%c0, %arg5, %37, %40] : memref<?x3x8x8xf32>
            %49 = arith.mulf %39, %48 : f32
            %50 = arith.addf %44, %49 : f32
            scf.yield %50 : f32
          } else {
            scf.yield %44 : f32
          }
          %46 = scf.if %31 -> (f32) {
            %48 = memref.load %arg0[%c0, %arg5, %42, %38] : memref<?x3x8x8xf32>
            %49 = arith.mulf %41, %48 : f32
            %50 = arith.addf %45, %49 : f32
            scf.yield %50 : f32
          } else {
            scf.yield %45 : f32
          }
          %47 = scf.if %33 -> (f32) {
            %48 = memref.load %arg0[%c0, %arg5, %42, %40] : memref<?x3x8x8xf32>
            %49 = arith.mulf %43, %48 : f32
            %50 = arith.addf %46, %49 : f32
            scf.yield %50 : f32
          } else {
            scf.yield %46 : f32
          }
          affine.store %47, %arg2[0, %arg5, %arg3, %arg4] : memref<?x3x6x6xf32>
        }
      }
    }
    return
  }
}
