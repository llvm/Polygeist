module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<3x3xf64>
    %alloca_1 = memref.alloca() : memref<3x3xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %arg2[%arg6 + %arg5 * 1125] : memref<?xf64>
        %1 = affine.load %arg2[%arg6 + %arg5 * 1125 + 125] : memref<?xf64>
        %2 = affine.load %arg2[%arg6 + %arg5 * 1125 + 250] : memref<?xf64>
        %3 = affine.load %arg2[%arg6 + %arg5 * 1125 + 375] : memref<?xf64>
        %4 = affine.load %arg2[%arg6 + %arg5 * 1125 + 500] : memref<?xf64>
        %5 = affine.load %arg2[%arg6 + %arg5 * 1125 + 625] : memref<?xf64>
        %6 = affine.load %arg2[%arg6 + %arg5 * 1125 + 750] : memref<?xf64>
        %7 = affine.load %arg2[%arg6 + %arg5 * 1125 + 875] : memref<?xf64>
        %8 = affine.load %arg2[%arg6 + %arg5 * 1125 + 1000] : memref<?xf64>
        %9 = arith.mulf %4, %8 : f64
        %10 = arith.mulf %5, %7 : f64
        %11 = arith.subf %9, %10 : f64
        %12 = arith.mulf %0, %11 : f64
        %13 = arith.mulf %3, %8 : f64
        %14 = arith.mulf %5, %6 : f64
        %15 = arith.subf %13, %14 : f64
        %16 = arith.mulf %1, %15 : f64
        %17 = arith.subf %12, %16 : f64
        %18 = arith.mulf %3, %7 : f64
        %19 = arith.mulf %4, %6 : f64
        %20 = arith.subf %18, %19 : f64
        %21 = arith.mulf %2, %20 : f64
        %22 = arith.addf %17, %21 : f64
        %23 = arith.divf %11, %22 : f64
        affine.store %23, %alloca_1[0, 0] : memref<3x3xf64>
        %24 = arith.mulf %2, %7 : f64
        %25 = arith.mulf %1, %8 : f64
        %26 = arith.subf %24, %25 : f64
        %27 = arith.divf %26, %22 : f64
        affine.store %27, %alloca_1[0, 1] : memref<3x3xf64>
        %28 = arith.mulf %1, %5 : f64
        %29 = arith.mulf %2, %4 : f64
        %30 = arith.subf %28, %29 : f64
        %31 = arith.divf %30, %22 : f64
        affine.store %31, %alloca_1[0, 2] : memref<3x3xf64>
        %32 = arith.subf %14, %13 : f64
        %33 = arith.divf %32, %22 : f64
        affine.store %33, %alloca_1[1, 0] : memref<3x3xf64>
        %34 = arith.mulf %0, %8 : f64
        %35 = arith.mulf %2, %6 : f64
        %36 = arith.subf %34, %35 : f64
        %37 = arith.divf %36, %22 : f64
        affine.store %37, %alloca_1[1, 1] : memref<3x3xf64>
        %38 = arith.mulf %2, %3 : f64
        %39 = arith.mulf %0, %5 : f64
        %40 = arith.subf %38, %39 : f64
        %41 = arith.divf %40, %22 : f64
        affine.store %41, %alloca_1[1, 2] : memref<3x3xf64>
        %42 = arith.divf %20, %22 : f64
        affine.store %42, %alloca_1[2, 0] : memref<3x3xf64>
        %43 = arith.mulf %1, %6 : f64
        %44 = arith.mulf %0, %7 : f64
        %45 = arith.subf %43, %44 : f64
        %46 = arith.divf %45, %22 : f64
        affine.store %46, %alloca_1[2, 1] : memref<3x3xf64>
        %47 = arith.mulf %0, %4 : f64
        %48 = arith.mulf %1, %3 : f64
        %49 = arith.subf %47, %48 : f64
        %50 = arith.divf %49, %22 : f64
        affine.store %50, %alloca_1[2, 2] : memref<3x3xf64>
        %51 = affine.load %arg4[%arg6 + %arg5 * 1125] : memref<?xf64>
        %52 = affine.load %arg4[%arg6 + %arg5 * 1125 + 500] : memref<?xf64>
        %53 = arith.addf %51, %52 : f64
        %54 = affine.load %arg4[%arg6 + %arg5 * 1125 + 1000] : memref<?xf64>
        %55 = arith.addf %53, %54 : f64
        %56 = affine.load %arg3[%arg6] : memref<?xf64>
        %57 = arith.mulf %56, %22 : f64
        %58 = affine.load %arg0[%arg6 + %arg5 * 125] : memref<?xf64>
        %59 = affine.load %arg1[%arg6 + %arg5 * 125] : memref<?xf64>
        %60 = arith.mulf %59, %cst : f64
        affine.for %arg7 = 0 to 3 {
          affine.for %arg8 = 0 to 3 {
            %61 = arith.index_cast %arg8 : index to i32
            %62 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %cst_0) -> (f64) {
              %69 = arith.index_cast %arg9 : index to i32
              %70 = arith.cmpi eq, %69, %61 : i32
              %71 = arith.extui %70 : i1 to i32
              %72 = arith.sitofp %71 : i32 to f64
              %73 = affine.load %alloca_1[%arg7, %arg9] : memref<3x3xf64>
              %74 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %arg10) -> (f64) {
                %75 = arith.index_cast %arg11 : index to i32
                %76 = affine.load %alloca_1[%arg7, %arg11] : memref<3x3xf64>
                %77 = arith.mulf %72, %76 : f64
                %78 = arith.cmpi eq, %75, %61 : i32
                %79 = arith.extui %78 : i1 to i32
                %80 = arith.sitofp %79 : i32 to f64
                %81 = arith.mulf %80, %73 : f64
                %82 = arith.addf %77, %81 : f64
                %83 = affine.load %arg4[%arg5 * 1125 + %arg6 + %arg9 * 375 + %arg11 * 125] : memref<?xf64>
                %84 = affine.load %arg4[%arg5 * 1125 + %arg6 + %arg11 * 375 + %arg9 * 125] : memref<?xf64>
                %85 = arith.addf %83, %84 : f64
                %86 = arith.mulf %82, %85 : f64
                %87 = arith.addf %arg12, %86 : f64
                affine.yield %87 : f64
              }
              affine.yield %74 : f64
            }
            %63 = affine.load %alloca_1[%arg7, %arg8] : memref<3x3xf64>
            %64 = arith.mulf %58, %63 : f64
            %65 = arith.mulf %64, %55 : f64
            %66 = arith.mulf %60, %62 : f64
            %67 = arith.addf %65, %66 : f64
            %68 = arith.mulf %57, %67 : f64
            affine.store %68, %alloca[%arg7, %arg8] : memref<3x3xf64>
          }
        }
        affine.for %arg7 = 0 to 3 {
          affine.for %arg8 = 0 to 3 {
            %61 = affine.load %alloca[%arg7, %arg8] : memref<3x3xf64>
            affine.store %61, %arg4[%arg5 * 1125 + %arg6 + %arg8 * 375 + %arg7 * 125] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}
