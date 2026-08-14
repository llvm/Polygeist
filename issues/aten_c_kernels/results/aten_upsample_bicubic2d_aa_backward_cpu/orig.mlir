module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_aa_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.571428597 : f32
    %cst_0 = arith.constant 6.250000e-01 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %cst_2 = arith.constant 2.500000e+00 : f32
    %cst_3 = arith.constant 1.500000e+00 : f32
    %cst_4 = arith.constant 2.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 4.000000e+00 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 40 {
      affine.store %cst_8, %arg1[%arg2] : memref<?xf32>
    }
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 7 {
        %0 = arith.index_cast %arg3 : index to i32
        %1 = arith.sitofp %0 : i32 to f32
        %2 = arith.addf %1, %cst_5 : f32
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.subf %3, %cst_5 : f32
        affine.for %arg4 = 0 to 8 {
          %5 = arith.index_cast %arg4 : index to i32
          %6 = arith.sitofp %5 : i32 to f32
          %7 = arith.addf %6, %cst_5 : f32
          %8 = arith.mulf %7, %cst_0 : f32
          %9 = arith.subf %8, %cst_5 : f32
          %10 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %cst_8) -> (f32) {
            %11 = arith.index_cast %arg5 : index to i32
            %12 = arith.sitofp %11 : i32 to f32
            %13 = arith.subf %4, %12 : f32
            %14 = arith.cmpf olt, %13, %cst_8 : f32
            %15 = scf.if %14 -> (f32) {
              %20 = arith.negf %13 : f32
              scf.yield %20 : f32
            } else {
              scf.yield %13 : f32
            }
            %16 = arith.cmpf olt, %15, %cst_4 : f32
            %17 = arith.cmpf olt, %15, %cst_6 : f32
            %18 = scf.if %16 -> (f32) {
              %20 = scf.if %17 -> (f32) {
                %21 = arith.mulf %15, %cst_3 : f32
                %22 = arith.subf %21, %cst_2 : f32
                %23 = arith.mulf %22, %15 : f32
                %24 = arith.mulf %23, %15 : f32
                %25 = arith.addf %24, %cst_6 : f32
                scf.yield %25 : f32
              } else {
                %21 = arith.mulf %15, %cst_1 : f32
                %22 = arith.addf %21, %cst_2 : f32
                %23 = arith.mulf %22, %15 : f32
                %24 = arith.subf %23, %cst_7 : f32
                %25 = arith.mulf %24, %15 : f32
                %26 = arith.addf %25, %cst_4 : f32
                scf.yield %26 : f32
              }
              scf.yield %20 : f32
            } else {
              scf.yield %cst_8 : f32
            }
            %19 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %arg6) -> (f32) {
              %20 = arith.index_cast %arg7 : index to i32
              %21 = arith.sitofp %20 : i32 to f32
              %22 = arith.subf %9, %21 : f32
              %23 = arith.cmpf olt, %22, %cst_8 : f32
              %24 = scf.if %23 -> (f32) {
                %30 = arith.negf %22 : f32
                scf.yield %30 : f32
              } else {
                scf.yield %22 : f32
              }
              %25 = arith.cmpf olt, %24, %cst_4 : f32
              %26 = arith.cmpf olt, %24, %cst_6 : f32
              %27 = scf.if %25 -> (f32) {
                %30 = scf.if %26 -> (f32) {
                  %31 = arith.mulf %24, %cst_3 : f32
                  %32 = arith.subf %31, %cst_2 : f32
                  %33 = arith.mulf %32, %24 : f32
                  %34 = arith.mulf %33, %24 : f32
                  %35 = arith.addf %34, %cst_6 : f32
                  scf.yield %35 : f32
                } else {
                  %31 = arith.mulf %24, %cst_1 : f32
                  %32 = arith.addf %31, %cst_2 : f32
                  %33 = arith.mulf %32, %24 : f32
                  %34 = arith.subf %33, %cst_7 : f32
                  %35 = arith.mulf %34, %24 : f32
                  %36 = arith.addf %35, %cst_4 : f32
                  scf.yield %36 : f32
                }
                scf.yield %30 : f32
              } else {
                scf.yield %cst_8 : f32
              }
              %28 = arith.mulf %18, %27 : f32
              %29 = arith.addf %arg8, %28 : f32
              affine.yield %29 : f32
            }
            affine.yield %19 : f32
          }
          affine.for %arg5 = 0 to 4 {
            %11 = arith.index_cast %arg5 : index to i32
            %12 = arith.sitofp %11 : i32 to f32
            %13 = arith.subf %4, %12 : f32
            %14 = arith.cmpf olt, %13, %cst_8 : f32
            %15 = scf.if %14 -> (f32) {
              %19 = arith.negf %13 : f32
              scf.yield %19 : f32
            } else {
              scf.yield %13 : f32
            }
            %16 = arith.cmpf olt, %15, %cst_4 : f32
            %17 = arith.cmpf olt, %15, %cst_6 : f32
            %18 = scf.if %16 -> (f32) {
              %19 = scf.if %17 -> (f32) {
                %20 = arith.mulf %15, %cst_3 : f32
                %21 = arith.subf %20, %cst_2 : f32
                %22 = arith.mulf %21, %15 : f32
                %23 = arith.mulf %22, %15 : f32
                %24 = arith.addf %23, %cst_6 : f32
                scf.yield %24 : f32
              } else {
                %20 = arith.mulf %15, %cst_1 : f32
                %21 = arith.addf %20, %cst_2 : f32
                %22 = arith.mulf %21, %15 : f32
                %23 = arith.subf %22, %cst_7 : f32
                %24 = arith.mulf %23, %15 : f32
                %25 = arith.addf %24, %cst_4 : f32
                scf.yield %25 : f32
              }
              scf.yield %19 : f32
            } else {
              scf.yield %cst_8 : f32
            }
            affine.for %arg6 = 0 to 5 {
              %19 = arith.index_cast %arg6 : index to i32
              %20 = arith.sitofp %19 : i32 to f32
              %21 = arith.subf %9, %20 : f32
              %22 = arith.cmpf olt, %21, %cst_8 : f32
              %23 = scf.if %22 -> (f32) {
                %33 = arith.negf %21 : f32
                scf.yield %33 : f32
              } else {
                scf.yield %21 : f32
              }
              %24 = arith.cmpf olt, %23, %cst_4 : f32
              %25 = arith.cmpf olt, %23, %cst_6 : f32
              %26 = scf.if %24 -> (f32) {
                %33 = scf.if %25 -> (f32) {
                  %34 = arith.mulf %23, %cst_3 : f32
                  %35 = arith.subf %34, %cst_2 : f32
                  %36 = arith.mulf %35, %23 : f32
                  %37 = arith.mulf %36, %23 : f32
                  %38 = arith.addf %37, %cst_6 : f32
                  scf.yield %38 : f32
                } else {
                  %34 = arith.mulf %23, %cst_1 : f32
                  %35 = arith.addf %34, %cst_2 : f32
                  %36 = arith.mulf %35, %23 : f32
                  %37 = arith.subf %36, %cst_7 : f32
                  %38 = arith.mulf %37, %23 : f32
                  %39 = arith.addf %38, %cst_4 : f32
                  scf.yield %39 : f32
                }
                scf.yield %33 : f32
              } else {
                scf.yield %cst_8 : f32
              }
              %27 = affine.load %arg0[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
              %28 = arith.mulf %27, %18 : f32
              %29 = arith.mulf %28, %26 : f32
              %30 = arith.divf %29, %10 : f32
              %31 = affine.load %arg1[%arg6 + %arg2 * 20 + %arg5 * 5] : memref<?xf32>
              %32 = arith.addf %31, %30 : f32
              affine.store %32, %arg1[%arg6 + %arg2 * 20 + %arg5 * 5] : memref<?xf32>
            }
          }
        }
      }
    }
    return
  }
}
