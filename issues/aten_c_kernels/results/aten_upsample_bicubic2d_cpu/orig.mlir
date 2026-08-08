module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 1.250000e+00 : f32
    %cst_1 = arith.constant 2.250000e+00 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %cst_3 = arith.constant -7.500000e-01 : f32
    %cst_4 = arith.constant 3.750000e+00 : f32
    %cst_5 = arith.constant 6.000000e+00 : f32
    %cst_6 = arith.constant 3.000000e+00 : f32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 8.000000e+00 : f32
    %cst_9 = arith.constant 5.000000e+00 : f32
    %cst_10 = arith.constant 7.000000e+00 : f32
    %cst_11 = arith.constant 4.000000e+00 : f32
    %cst_12 = arith.constant 5.000000e-01 : f32
    %0 = llvm.mlir.undef : f32
    affine.for %arg2 = 0 to 2 {
      %1 = arith.index_cast %arg2 : index to i32
      %2 = arith.muli %1, %c4_i32 : i32
      affine.for %arg3 = 0 to 7 {
        %3 = arith.index_cast %arg3 : index to i32
        %4 = arith.sitofp %3 : i32 to f32
        %5 = arith.addf %4, %cst_12 : f32
        %6 = arith.mulf %5, %cst_11 : f32
        %7 = arith.divf %6, %cst_10 : f32
        %8 = arith.subf %7, %cst_12 : f32
        %9 = arith.fptosi %8 : f32 to i32
        %10 = arith.cmpf olt, %8, %cst_7 : f32
        %11 = arith.sitofp %9 : i32 to f32
        %12 = arith.cmpf une, %8, %11 : f32
        %13 = arith.andi %10, %12 : i1
        %14 = scf.if %13 -> (i32) {
          %15 = arith.addi %9, %c-1_i32 : i32
          scf.yield %15 : i32
        } else {
          scf.yield %9 : i32
        }
        affine.for %arg4 = 0 to 8 {
          %15 = arith.index_cast %arg4 : index to i32
          %16 = arith.sitofp %15 : i32 to f32
          %17 = arith.addf %16, %cst_12 : f32
          %18 = arith.mulf %17, %cst_9 : f32
          %19 = arith.divf %18, %cst_8 : f32
          %20 = arith.subf %19, %cst_12 : f32
          %21 = arith.fptosi %20 : f32 to i32
          %22 = arith.cmpf olt, %20, %cst_7 : f32
          %23 = arith.sitofp %21 : i32 to f32
          %24 = arith.cmpf une, %20, %23 : f32
          %25 = arith.andi %22, %24 : i1
          %26 = scf.if %25 -> (i32) {
            %28 = arith.addi %21, %c-1_i32 : i32
            scf.yield %28 : i32
          } else {
            scf.yield %21 : i32
          }
          %27 = affine.for %arg5 = -1 to 3 iter_args(%arg6 = %cst_7) -> (f32) {
            %28 = arith.index_cast %arg5 : index to i32
            %29 = arith.addi %14, %28 : i32
            %30 = arith.cmpi slt, %29, %c0_i32 : i32
            %31 = arith.select %30, %c0_i32, %29 : i32
            %32 = arith.sitofp %29 : i32 to f32
            %33 = arith.subf %8, %32 : f32
            %34 = scf.if %30 -> (i1) {
              scf.yield %false : i1
            } else {
              %50 = arith.cmpi sge, %29, %c4_i32 : i32
              scf.yield %50 : i1
            }
            %35 = arith.select %34, %c3_i32, %31 : i32
            %36 = arith.addi %2, %35 : i32
            %37 = arith.muli %36, %c5_i32 : i32
            %38 = arith.cmpf olt, %33, %cst_7 : f32
            %39 = scf.if %38 -> (f32) {
              %50 = arith.negf %33 : f32
              scf.yield %50 : f32
            } else {
              scf.yield %33 : f32
            }
            %40 = arith.cmpf olt, %39, %cst : f32
            %41 = arith.xori %40, %true : i1
            %42 = scf.if %40 -> (f32) {
              %50 = arith.mulf %39, %cst_0 : f32
              %51 = arith.subf %50, %cst_1 : f32
              %52 = arith.mulf %51, %39 : f32
              %53 = arith.mulf %52, %39 : f32
              %54 = arith.addf %53, %cst : f32
              scf.yield %54 : f32
            } else {
              scf.yield %0 : f32
            }
            %43 = arith.cmpf olt, %39, %cst_2 : f32
            %44 = arith.andi %43, %41 : i1
            %45 = arith.xori %44, %true : i1
            %46 = arith.andi %45, %41 : i1
            %47 = scf.if %44 -> (f32) {
              %50 = arith.mulf %39, %cst_3 : f32
              %51 = arith.addf %50, %cst_4 : f32
              %52 = arith.mulf %51, %39 : f32
              %53 = arith.subf %52, %cst_5 : f32
              %54 = arith.mulf %53, %39 : f32
              %55 = arith.addf %54, %cst_6 : f32
              scf.yield %55 : f32
            } else {
              scf.yield %42 : f32
            }
            %48 = arith.select %46, %cst_7, %47 : f32
            %49 = affine.for %arg7 = -1 to 3 iter_args(%arg8 = %arg6) -> (f32) {
              %50 = arith.index_cast %arg7 : index to i32
              %51 = arith.addi %26, %50 : i32
              %52 = arith.cmpi slt, %51, %c0_i32 : i32
              %53 = arith.select %52, %c0_i32, %51 : i32
              %54 = scf.if %52 -> (i1) {
                scf.yield %false : i1
              } else {
                %75 = arith.cmpi sge, %51, %c5_i32 : i32
                scf.yield %75 : i1
              }
              %55 = arith.select %54, %c4_i32, %53 : i32
              %56 = arith.addi %37, %55 : i32
              %57 = arith.index_cast %56 : i32 to index
              %58 = memref.load %arg0[%57] : memref<?xf32>
              %59 = arith.mulf %58, %48 : f32
              %60 = arith.sitofp %51 : i32 to f32
              %61 = arith.subf %20, %60 : f32
              %62 = arith.cmpf olt, %61, %cst_7 : f32
              %63 = scf.if %62 -> (f32) {
                %75 = arith.negf %61 : f32
                scf.yield %75 : f32
              } else {
                scf.yield %61 : f32
              }
              %64 = arith.cmpf olt, %63, %cst : f32
              %65 = arith.xori %64, %true : i1
              %66 = scf.if %64 -> (f32) {
                %75 = arith.mulf %63, %cst_0 : f32
                %76 = arith.subf %75, %cst_1 : f32
                %77 = arith.mulf %76, %63 : f32
                %78 = arith.mulf %77, %63 : f32
                %79 = arith.addf %78, %cst : f32
                scf.yield %79 : f32
              } else {
                scf.yield %0 : f32
              }
              %67 = arith.cmpf olt, %63, %cst_2 : f32
              %68 = arith.andi %67, %65 : i1
              %69 = arith.xori %68, %true : i1
              %70 = arith.andi %69, %65 : i1
              %71 = scf.if %68 -> (f32) {
                %75 = arith.mulf %63, %cst_3 : f32
                %76 = arith.addf %75, %cst_4 : f32
                %77 = arith.mulf %76, %63 : f32
                %78 = arith.subf %77, %cst_5 : f32
                %79 = arith.mulf %78, %63 : f32
                %80 = arith.addf %79, %cst_6 : f32
                scf.yield %80 : f32
              } else {
                scf.yield %66 : f32
              }
              %72 = arith.select %70, %cst_7, %71 : f32
              %73 = arith.mulf %59, %72 : f32
              %74 = arith.addf %arg8, %73 : f32
              affine.yield %74 : f32
            }
            affine.yield %49 : f32
          }
          affine.store %27, %arg1[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
        }
      }
    }
    return
  }
}
