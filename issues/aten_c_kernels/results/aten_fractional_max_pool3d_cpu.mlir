module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fractional_max_pool3d_cpu(%arg0: memref<?x2x8x9x10xf32>, %arg1: memref<?x2x3xf32>, %arg2: memref<?x2x3x4x5xf32>, %arg3: memref<?x2x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 6.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %cst_3 = arith.constant 4.000000e+00 : f32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_4 = arith.constant -3.40282347E+38 : f32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 3 {
        %0 = arith.index_cast %arg5 : index to i32
        %1 = arith.sitofp %0 : i32 to f32
        affine.for %arg6 = 0 to 4 {
          %2 = arith.index_cast %arg6 : index to i32
          %3 = arith.sitofp %2 : i32 to f32
          affine.for %arg7 = 0 to 5 {
            %4 = arith.index_cast %arg7 : index to i32
            %5 = affine.load %arg1[0, %arg4, 0] : memref<?x2x3xf32>
            %6 = arith.addf %1, %5 : f32
            %7 = arith.mulf %6, %cst_0 : f32
            %8 = arith.divf %7, %cst : f32
            %9 = arith.fptosi %8 : f32 to i32
            %10 = affine.load %arg1[0, %arg4, 1] : memref<?x2x3xf32>
            %11 = arith.addf %3, %10 : f32
            %12 = arith.mulf %11, %cst_0 : f32
            %13 = arith.divf %12, %cst_1 : f32
            %14 = arith.fptosi %13 : f32 to i32
            %15 = arith.sitofp %4 : i32 to f32
            %16 = affine.load %arg1[0, %arg4, 2] : memref<?x2x3xf32>
            %17 = arith.addf %15, %16 : f32
            %18 = arith.mulf %17, %cst_2 : f32
            %19 = arith.divf %18, %cst_3 : f32
            %20 = arith.fptosi %19 : f32 to i32
            %21 = arith.cmpi sgt, %9, %c6_i32 : i32
            %22 = arith.select %21, %c6_i32, %9 : i32
            %23 = arith.cmpi sgt, %14, %c6_i32 : i32
            %24 = arith.select %23, %c6_i32, %14 : i32
            %25 = arith.cmpi sgt, %20, %c7_i32 : i32
            %26 = arith.select %25, %c7_i32, %20 : i32
            %27:2 = affine.for %arg8 = 0 to 2 iter_args(%arg9 = %c0_i32, %arg10 = %cst_4) -> (i32, f32) {
              %28 = arith.index_cast %arg8 : index to i32
              %29 = arith.addi %22, %28 : i32
              %30 = arith.index_cast %29 : i32 to index
              %31 = arith.muli %29, %c9_i32 : i32
              %32 = arith.addi %31, %24 : i32
              %33:2 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (i32, f32) {
                %34 = arith.index_cast %arg11 : index to i32
                %35 = arith.addi %24, %34 : i32
                %36 = arith.index_cast %35 : i32 to index
                %37 = arith.addi %32, %34 : i32
                %38 = arith.muli %37, %c10_i32 : i32
                %39 = arith.addi %38, %26 : i32
                %40:2 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %arg12, %arg16 = %arg13) -> (i32, f32) {
                  %41 = arith.index_cast %arg14 : index to i32
                  %42 = arith.addi %26, %41 : i32
                  %43 = arith.index_cast %42 : i32 to index
                  %44 = memref.load %arg0[%c0, %arg4, %30, %36, %43] : memref<?x2x8x9x10xf32>
                  %45 = arith.cmpf ogt, %44, %arg16 : f32
                  %46:2 = scf.if %45 -> (i32, f32) {
                    %47 = memref.load %arg0[%c0, %arg4, %30, %36, %43] : memref<?x2x8x9x10xf32>
                    %48 = arith.addi %39, %41 : i32
                    scf.yield %48, %47 : i32, f32
                  } else {
                    scf.yield %arg15, %arg16 : i32, f32
                  }
                  affine.yield %46#0, %46#1 : i32, f32
                }
                affine.yield %40#0, %40#1 : i32, f32
              }
              affine.yield %33#0, %33#1 : i32, f32
            }
            affine.store %27#1, %arg2[0, %arg4, %arg5, %arg6, %arg7] : memref<?x2x3x4x5xf32>
            affine.store %27#0, %arg3[0, %arg4, %arg5, %arg6, %arg7] : memref<?x2x3x4x5xi32>
          }
        }
      }
    }
    return
  }
}
