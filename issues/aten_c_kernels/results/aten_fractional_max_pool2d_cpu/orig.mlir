module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fractional_max_pool2d_cpu(%arg0: memref<?x3x9x10xf32>, %arg1: memref<?x3x2xf32>, %arg2: memref<?x3x4x5xf32>, %arg3: memref<?x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %cst_1 = arith.constant 7.000000e+00 : f32
    %cst_2 = arith.constant 4.000000e+00 : f32
    %c10_i32 = arith.constant 10 : i32
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 4 {
          %0 = arith.index_cast %arg6 : index to i32
          %1 = arith.sitofp %0 : i32 to f32
          affine.for %arg7 = 0 to 5 {
            %2 = arith.index_cast %arg7 : index to i32
            %3 = affine.load %arg1[%arg4, %arg5, 0] : memref<?x3x2xf32>
            %4 = arith.addf %1, %3 : f32
            %5 = arith.mulf %4, %cst : f32
            %6 = arith.divf %5, %cst_0 : f32
            %7 = arith.fptosi %6 : f32 to i32
            %8 = arith.sitofp %2 : i32 to f32
            %9 = affine.load %arg1[%arg4, %arg5, 1] : memref<?x3x2xf32>
            %10 = arith.addf %8, %9 : f32
            %11 = arith.mulf %10, %cst_1 : f32
            %12 = arith.divf %11, %cst_2 : f32
            %13 = arith.fptosi %12 : f32 to i32
            %14 = arith.cmpi sgt, %7, %c6_i32 : i32
            %15 = arith.select %14, %c6_i32, %7 : i32
            %16 = arith.cmpi sgt, %13, %c7_i32 : i32
            %17 = arith.select %16, %c7_i32, %13 : i32
            %18:2 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %c0_i32, %arg10 = %cst_3) -> (i32, f32) {
              %19 = arith.index_cast %arg8 : index to i32
              %20 = arith.addi %15, %19 : i32
              %21 = arith.index_cast %20 : i32 to index
              %22 = arith.muli %20, %c10_i32 : i32
              %23 = arith.addi %22, %17 : i32
              %24:2 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (i32, f32) {
                %25 = arith.index_cast %arg11 : index to i32
                %26 = arith.addi %17, %25 : i32
                %27 = arith.index_cast %26 : i32 to index
                %28 = memref.load %arg0[%arg4, %arg5, %21, %27] : memref<?x3x9x10xf32>
                %29 = arith.cmpf ogt, %28, %arg13 : f32
                %30:2 = scf.if %29 -> (i32, f32) {
                  %31 = memref.load %arg0[%arg4, %arg5, %21, %27] : memref<?x3x9x10xf32>
                  %32 = arith.addi %23, %25 : i32
                  scf.yield %32, %31 : i32, f32
                } else {
                  scf.yield %arg12, %arg13 : i32, f32
                }
                affine.yield %30#0, %30#1 : i32, f32
              }
              affine.yield %24#0, %24#1 : i32, f32
            }
            affine.store %18#1, %arg2[%arg4, %arg5, %arg6, %arg7] : memref<?x3x4x5xf32>
            affine.store %18#0, %arg3[%arg4, %arg5, %arg6, %arg7] : memref<?x3x4x5xi32>
          }
        }
      }
    }
    return
  }
}
