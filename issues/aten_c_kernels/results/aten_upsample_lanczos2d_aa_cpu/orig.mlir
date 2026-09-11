module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_lanczos2d_aa_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.571428597 : f32
    %cst_0 = arith.constant 6.250000e-01 : f32
    %cst_1 = arith.constant 3.28986812 : f32
    %cst_2 = arith.constant 3.14159274 : f32
    %cst_3 = arith.constant 3.000000e+00 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
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
          %10 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %cst_4) -> (f32) {
            %13 = arith.index_cast %arg5 : index to i32
            %14 = arith.sitofp %13 : i32 to f32
            %15 = arith.subf %4, %14 : f32
            %16 = arith.cmpf olt, %15, %cst_4 : f32
            %17 = scf.if %16 -> (f32) {
              %25 = arith.negf %15 : f32
              scf.yield %25 : f32
            } else {
              scf.yield %15 : f32
            }
            %18 = arith.cmpf olt, %17, %cst_3 : f32
            %19 = arith.cmpf oeq, %17, %cst_4 : f32
            %20 = arith.mulf %17, %cst_2 : f32
            %21 = arith.divf %20, %cst_3 : f32
            %22 = arith.mulf %17, %cst_1 : f32
            %23 = arith.mulf %22, %17 : f32
            %24 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %arg6) -> (f32) {
              %25 = arith.index_cast %arg7 : index to i32
              %26 = arith.sitofp %25 : i32 to f32
              %27 = arith.subf %9, %26 : f32
              %28 = arith.cmpf olt, %27, %cst_4 : f32
              %29 = scf.if %28 -> (f32) {
                %36 = arith.negf %27 : f32
                scf.yield %36 : f32
              } else {
                scf.yield %27 : f32
              }
              %30 = scf.if %18 -> (f32) {
                %36 = scf.if %19 -> (f32) {
                  scf.yield %cst_6 : f32
                } else {
                  %37 = func.call @sinf(%20) : (f32) -> f32
                  %38 = func.call @sinf(%21) : (f32) -> f32
                  %39 = arith.mulf %37, %38 : f32
                  %40 = arith.divf %39, %23 : f32
                  scf.yield %40 : f32
                }
                scf.yield %36 : f32
              } else {
                scf.yield %cst_4 : f32
              }
              %31 = arith.cmpf olt, %29, %cst_3 : f32
              %32 = arith.cmpf oeq, %29, %cst_4 : f32
              %33 = scf.if %31 -> (f32) {
                %36 = scf.if %32 -> (f32) {
                  scf.yield %cst_6 : f32
                } else {
                  %37 = arith.mulf %29, %cst_2 : f32
                  %38 = func.call @sinf(%37) : (f32) -> f32
                  %39 = arith.divf %37, %cst_3 : f32
                  %40 = func.call @sinf(%39) : (f32) -> f32
                  %41 = arith.mulf %38, %40 : f32
                  %42 = arith.mulf %29, %cst_1 : f32
                  %43 = arith.mulf %42, %29 : f32
                  %44 = arith.divf %41, %43 : f32
                  scf.yield %44 : f32
                }
                scf.yield %36 : f32
              } else {
                scf.yield %cst_4 : f32
              }
              %34 = arith.mulf %30, %33 : f32
              %35 = arith.addf %arg8, %34 : f32
              affine.yield %35 : f32
            }
            affine.yield %24 : f32
          }
          %11 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %cst_4) -> (f32) {
            %13 = arith.index_cast %arg5 : index to i32
            %14 = arith.sitofp %13 : i32 to f32
            %15 = arith.subf %4, %14 : f32
            %16 = arith.cmpf olt, %15, %cst_4 : f32
            %17 = scf.if %16 -> (f32) {
              %25 = arith.negf %15 : f32
              scf.yield %25 : f32
            } else {
              scf.yield %15 : f32
            }
            %18 = arith.cmpf olt, %17, %cst_3 : f32
            %19 = arith.cmpf oeq, %17, %cst_4 : f32
            %20 = arith.mulf %17, %cst_2 : f32
            %21 = arith.divf %20, %cst_3 : f32
            %22 = arith.mulf %17, %cst_1 : f32
            %23 = arith.mulf %22, %17 : f32
            %24 = affine.for %arg7 = 0 to 5 iter_args(%arg8 = %arg6) -> (f32) {
              %25 = arith.index_cast %arg7 : index to i32
              %26 = arith.sitofp %25 : i32 to f32
              %27 = arith.subf %9, %26 : f32
              %28 = arith.cmpf olt, %27, %cst_4 : f32
              %29 = scf.if %28 -> (f32) {
                %38 = arith.negf %27 : f32
                scf.yield %38 : f32
              } else {
                scf.yield %27 : f32
              }
              %30 = scf.if %18 -> (f32) {
                %38 = scf.if %19 -> (f32) {
                  scf.yield %cst_6 : f32
                } else {
                  %39 = func.call @sinf(%20) : (f32) -> f32
                  %40 = func.call @sinf(%21) : (f32) -> f32
                  %41 = arith.mulf %39, %40 : f32
                  %42 = arith.divf %41, %23 : f32
                  scf.yield %42 : f32
                }
                scf.yield %38 : f32
              } else {
                scf.yield %cst_4 : f32
              }
              %31 = arith.cmpf olt, %29, %cst_3 : f32
              %32 = arith.cmpf oeq, %29, %cst_4 : f32
              %33 = scf.if %31 -> (f32) {
                %38 = scf.if %32 -> (f32) {
                  scf.yield %cst_6 : f32
                } else {
                  %39 = arith.mulf %29, %cst_2 : f32
                  %40 = func.call @sinf(%39) : (f32) -> f32
                  %41 = arith.divf %39, %cst_3 : f32
                  %42 = func.call @sinf(%41) : (f32) -> f32
                  %43 = arith.mulf %40, %42 : f32
                  %44 = arith.mulf %29, %cst_1 : f32
                  %45 = arith.mulf %44, %29 : f32
                  %46 = arith.divf %43, %45 : f32
                  scf.yield %46 : f32
                }
                scf.yield %38 : f32
              } else {
                scf.yield %cst_4 : f32
              }
              %34 = affine.load %arg0[%arg7 + %arg2 * 20 + %arg5 * 5] : memref<?xf32>
              %35 = arith.mulf %34, %30 : f32
              %36 = arith.mulf %35, %33 : f32
              %37 = arith.addf %arg8, %36 : f32
              affine.yield %37 : f32
            }
            affine.yield %24 : f32
          }
          %12 = arith.divf %11, %10 : f32
          affine.store %12, %arg1[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
        }
      }
    }
    return
  }
  func.func private @sinf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
