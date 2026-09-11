module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_3d_backward_cpu(%arg0: memref<?x2x4x5x6xf32>, %arg1: memref<?x4x5x6x3xf32>, %arg2: memref<?x2x6x7x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 5.000000e+00 : f32
    %cst_3 = arith.constant 7.000000e+00 : f32
    %cst_4 = arith.constant 6.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x4x5x6x3xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x2x4x5x6xf32>
    %2 = "polygeist.memref2pointer"(%arg2) : (memref<?x2x6x7x8xf32>) -> !llvm.ptr
    affine.for %arg3 = 0 to 672 {
      %3 = arith.index_cast %arg3 : index to i32
      %4 = llvm.getelementptr %2[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %4 : f32, !llvm.ptr
    }
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 6 {
          %extracted = tensor.extract %0[%c0, %arg3, %arg4, %arg5, %c0] : tensor<?x4x5x6x3xf32>
          %3 = arith.addf %extracted, %cst_0 : f32
          %4 = arith.mulf %3, %cst_1 : f32
          %5 = arith.mulf %4, %cst_3 : f32
          %extracted_5 = tensor.extract %0[%c0, %arg3, %arg4, %arg5, %c1] : tensor<?x4x5x6x3xf32>
          %6 = arith.addf %extracted_5, %cst_0 : f32
          %7 = arith.mulf %6, %cst_1 : f32
          %8 = arith.mulf %7, %cst_4 : f32
          %extracted_6 = tensor.extract %0[%c0, %arg3, %arg4, %arg5, %c2] : tensor<?x4x5x6x3xf32>
          %9 = arith.addf %extracted_6, %cst_0 : f32
          %10 = arith.mulf %9, %cst_1 : f32
          %11 = arith.mulf %10, %cst_2 : f32
          %12 = arith.fptosi %5 : f32 to i32
          %13 = arith.fptosi %8 : f32 to i32
          %14 = arith.fptosi %11 : f32 to i32
          %15 = arith.sitofp %12 : i32 to f32
          %16 = arith.subf %5, %15 : f32
          %17 = arith.sitofp %13 : i32 to f32
          %18 = arith.subf %8, %17 : f32
          %19 = arith.sitofp %14 : i32 to f32
          %20 = arith.subf %11, %19 : f32
          %21 = arith.subf %cst_0, %16 : f32
          %22 = arith.subf %cst_0, %18 : f32
          %23 = arith.subf %cst_0, %20 : f32
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %24 = arith.index_cast %arg7 : index to i32
              %25 = arith.addi %14, %24 : i32
              %26 = arith.cmpi sge, %25, %c0_i32 : i32
              %27 = arith.cmpi slt, %25, %c6_i32 : i32
              %28 = arith.index_cast %25 : i32 to index
              %29 = arith.cmpi ne, %24, %c0_i32 : i32
              %30 = arith.select %29, %20, %23 : f32
              affine.for %arg8 = 0 to 2 {
                %31 = arith.index_cast %arg8 : index to i32
                %32 = arith.addi %13, %31 : i32
                %33 = arith.cmpi sge, %32, %c0_i32 : i32
                %34 = arith.cmpi slt, %32, %c7_i32 : i32
                %35 = arith.index_cast %32 : i32 to index
                %36 = arith.cmpi ne, %31, %c0_i32 : i32
                %37 = arith.select %36, %18, %22 : f32
                affine.for %arg9 = 0 to 2 {
                  %38 = arith.index_cast %arg9 : index to i32
                  %39 = arith.addi %12, %38 : i32
                  %40 = arith.cmpi sge, %39, %c0_i32 : i32
                  %41 = arith.cmpi slt, %39, %c8_i32 : i32
                  %42 = arith.andi %40, %41 : i1
                  %43 = arith.andi %34, %42 : i1
                  %44 = arith.andi %33, %43 : i1
                  %45 = arith.andi %27, %44 : i1
                  %46 = arith.andi %26, %45 : i1
                  scf.if %46 {
                    %47 = arith.index_cast %39 : i32 to index
                    %extracted_7 = tensor.extract %1[%c0, %arg6, %arg3, %arg4, %arg5] : tensor<?x2x4x5x6xf32>
                    %48 = arith.mulf %extracted_7, %30 : f32
                    %49 = arith.mulf %48, %37 : f32
                    %50 = arith.cmpi ne, %38, %c0_i32 : i32
                    %51 = arith.select %50, %16, %21 : f32
                    %52 = arith.mulf %49, %51 : f32
                    %53 = memref.load %arg2[%c0, %arg6, %28, %35, %47] : memref<?x2x6x7x8xf32>
                    %54 = arith.addf %53, %52 : f32
                    memref.store %54, %arg2[%c0, %arg6, %28, %35, %47] : memref<?x2x6x7x8xf32>
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

