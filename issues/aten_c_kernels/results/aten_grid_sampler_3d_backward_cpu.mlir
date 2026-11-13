module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_3d_backward_cpu(%arg0: memref<?x2x4x5x6xf32>, %arg1: memref<?x4x5x6x3xf32>, %arg2: memref<?x2x6x7x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 7.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = "polygeist.memref2pointer"(%arg2) : (memref<?x2x6x7x8xf32>) -> !llvm.ptr
    affine.for %arg3 = 0 to 672 {
      %1 = arith.index_cast %arg3 : index to i32
      %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_4, %2 : f32, !llvm.ptr
    }
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 6 {
          %1 = affine.load %arg1[0, %arg3, %arg4, %arg5, 0] : memref<?x4x5x6x3xf32>
          %2 = arith.addf %1, %cst_3 : f32
          %3 = arith.mulf %2, %cst_2 : f32
          %4 = arith.mulf %3, %cst_0 : f32
          %5 = affine.load %arg1[0, %arg3, %arg4, %arg5, 1] : memref<?x4x5x6x3xf32>
          %6 = arith.addf %5, %cst_3 : f32
          %7 = arith.mulf %6, %cst_2 : f32
          %8 = arith.mulf %7, %cst : f32
          %9 = affine.load %arg1[0, %arg3, %arg4, %arg5, 2] : memref<?x4x5x6x3xf32>
          %10 = arith.addf %9, %cst_3 : f32
          %11 = arith.mulf %10, %cst_2 : f32
          %12 = arith.mulf %11, %cst_1 : f32
          %13 = arith.fptosi %4 : f32 to i32
          %14 = arith.fptosi %8 : f32 to i32
          %15 = arith.fptosi %12 : f32 to i32
          %16 = arith.sitofp %13 : i32 to f32
          %17 = arith.subf %4, %16 : f32
          %18 = arith.sitofp %14 : i32 to f32
          %19 = arith.subf %8, %18 : f32
          %20 = arith.sitofp %15 : i32 to f32
          %21 = arith.subf %12, %20 : f32
          %22 = arith.subf %cst_3, %17 : f32
          %23 = arith.subf %cst_3, %19 : f32
          %24 = arith.subf %cst_3, %21 : f32
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %25 = arith.index_cast %arg7 : index to i32
              %26 = arith.addi %15, %25 : i32
              %27 = arith.cmpi sge, %26, %c0_i32 : i32
              %28 = arith.cmpi slt, %26, %c6_i32 : i32
              %29 = arith.index_cast %26 : i32 to index
              %30 = arith.cmpi ne, %25, %c0_i32 : i32
              %31 = arith.select %30, %21, %24 : f32
              affine.for %arg8 = 0 to 2 {
                %32 = arith.index_cast %arg8 : index to i32
                %33 = arith.addi %14, %32 : i32
                %34 = arith.cmpi sge, %33, %c0_i32 : i32
                %35 = arith.cmpi slt, %33, %c7_i32 : i32
                %36 = arith.index_cast %33 : i32 to index
                %37 = arith.cmpi ne, %32, %c0_i32 : i32
                %38 = arith.select %37, %19, %23 : f32
                affine.for %arg9 = 0 to 2 {
                  %39 = arith.index_cast %arg9 : index to i32
                  %40 = arith.addi %13, %39 : i32
                  %41 = arith.cmpi sge, %40, %c0_i32 : i32
                  %42 = arith.cmpi slt, %40, %c8_i32 : i32
                  %43 = arith.andi %41, %42 : i1
                  %44 = arith.andi %35, %43 : i1
                  %45 = arith.andi %34, %44 : i1
                  %46 = arith.andi %28, %45 : i1
                  %47 = arith.andi %27, %46 : i1
                  scf.if %47 {
                    %48 = arith.index_cast %40 : i32 to index
                    %49 = affine.load %arg0[0, %arg6, %arg3, %arg4, %arg5] : memref<?x2x4x5x6xf32>
                    %50 = arith.mulf %49, %31 : f32
                    %51 = arith.mulf %50, %38 : f32
                    %52 = arith.cmpi ne, %39, %c0_i32 : i32
                    %53 = arith.select %52, %17, %22 : f32
                    %54 = arith.mulf %51, %53 : f32
                    %55 = memref.load %arg2[%c0, %arg6, %29, %36, %48] : memref<?x2x6x7x8xf32>
                    %56 = arith.addf %55, %54 : f32
                    memref.store %56, %arg2[%c0, %arg6, %29, %36, %48] : memref<?x2x6x7x8xf32>
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
