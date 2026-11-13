#map = affine_map<(d0, d1, d2) -> (d0 + d1 * 56 + d2 * 8)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant 7.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e+00 : f32
    %cst_3 = arith.constant 8.000000e+00 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c5_i32 = arith.constant 5 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst_5 = arith.constant 3.000000e+00 : f32
    %cst_6 = arith.constant 6.000000e+00 : f32
    %cst_7 = arith.constant 3.750000e+00 : f32
    %cst_8 = arith.constant -7.500000e-01 : f32
    %cst_9 = arith.constant 2.000000e+00 : f32
    %cst_10 = arith.constant 2.250000e+00 : f32
    %cst_11 = arith.constant 1.250000e+00 : f32
    %cst_12 = arith.constant 1.000000e+00 : f32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = llvm.mlir.undef : f32
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %0) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.addf %9, %cst : f32
        %11 = arith.mulf %10, %cst_0 : f32
        %12 = arith.divf %11, %cst_1 : f32
        %13 = arith.subf %12, %cst : f32
        %14 = arith.fptosi %13 : f32 to i32
        %15 = arith.cmpf olt, %13, %cst_4 : f32
        %16 = arith.sitofp %14 : i32 to f32
        %17 = arith.cmpf une, %13, %16 : f32
        %18 = arith.andi %15, %17 : i1
        %19 = arith.addi %14, %c-1_i32 : i32
        %20 = arith.select %18, %19, %14 : i32
        %21 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %22 = arith.index_cast %arg6 : index to i32
          %23 = arith.sitofp %22 : i32 to f32
          %24 = arith.addf %23, %cst : f32
          %25 = arith.mulf %24, %cst_2 : f32
          %26 = arith.divf %25, %cst_3 : f32
          %27 = arith.subf %26, %cst : f32
          %28 = arith.fptosi %27 : f32 to i32
          %29 = arith.cmpf olt, %27, %cst_4 : f32
          %30 = arith.sitofp %28 : i32 to f32
          %31 = arith.cmpf une, %27, %30 : f32
          %32 = arith.andi %29, %31 : i1
          %33 = arith.addi %28, %c-1_i32 : i32
          %34 = arith.select %32, %33, %28 : i32
          %35 = affine.apply #map(%arg6, %arg2, %arg4)
          %inserted = tensor.insert %cst_4 into %arg7[%35] : tensor<?xf32>
          %36 = affine.for %arg8 = -1 to 3 iter_args(%arg9 = %inserted) -> (tensor<?xf32>) {
            %37 = arith.index_cast %arg8 : index to i32
            %38 = arith.addi %20, %37 : i32
            %39 = arith.cmpi slt, %38, %c0_i32 : i32
            %40 = arith.select %39, %c0_i32, %38 : i32
            %41 = arith.sitofp %38 : i32 to f32
            %42 = arith.subf %13, %41 : f32
            %43 = arith.cmpi sge, %38, %c4_i32 : i32
            %44 = arith.select %39, %false, %43 : i1
            %45 = arith.select %44, %c3_i32, %40 : i32
            %46 = arith.addi %6, %45 : i32
            %47 = arith.muli %46, %c5_i32 : i32
            %48 = arith.cmpf olt, %42, %cst_4 : f32
            %49 = arith.negf %42 : f32
            %50 = arith.select %48, %49, %42 : f32
            %51 = arith.cmpf olt, %50, %cst_12 : f32
            %52 = arith.xori %51, %true : i1
            %53 = arith.mulf %50, %cst_11 : f32
            %54 = arith.subf %53, %cst_10 : f32
            %55 = arith.mulf %54, %50 : f32
            %56 = arith.mulf %55, %50 : f32
            %57 = arith.addf %56, %cst_12 : f32
            %58 = arith.select %51, %57, %2 : f32
            %59 = arith.cmpf olt, %50, %cst_9 : f32
            %60 = arith.andi %59, %52 : i1
            %61 = arith.xori %60, %true : i1
            %62 = arith.andi %61, %52 : i1
            %63 = arith.mulf %50, %cst_8 : f32
            %64 = arith.addf %63, %cst_7 : f32
            %65 = arith.mulf %64, %50 : f32
            %66 = arith.subf %65, %cst_6 : f32
            %67 = arith.mulf %66, %50 : f32
            %68 = arith.addf %67, %cst_5 : f32
            %69 = arith.select %60, %68, %58 : f32
            %70 = arith.select %62, %cst_4, %69 : f32
            %71 = affine.for %arg10 = -1 to 3 iter_args(%arg11 = %arg9) -> (tensor<?xf32>) {
              %72 = affine.apply #map(%arg6, %arg2, %arg4)
              %extracted = tensor.extract %arg11[%72] : tensor<?xf32>
              %73 = arith.index_cast %arg10 : index to i32
              %74 = arith.addi %34, %73 : i32
              %75 = arith.cmpi slt, %74, %c0_i32 : i32
              %76 = arith.select %75, %c0_i32, %74 : i32
              %77 = arith.cmpi sge, %74, %c5_i32 : i32
              %78 = arith.select %75, %false, %77 : i1
              %79 = arith.select %78, %c4_i32, %76 : i32
              %80 = arith.addi %47, %79 : i32
              %81 = arith.index_cast %80 : i32 to index
              %extracted_13 = tensor.extract %1[%81] : tensor<?xf32>
              %82 = arith.mulf %extracted_13, %70 : f32
              %83 = arith.sitofp %74 : i32 to f32
              %84 = arith.subf %27, %83 : f32
              %85 = arith.cmpf olt, %84, %cst_4 : f32
              %86 = arith.negf %84 : f32
              %87 = arith.select %85, %86, %84 : f32
              %88 = arith.cmpf olt, %87, %cst_12 : f32
              %89 = arith.xori %88, %true : i1
              %90 = arith.mulf %87, %cst_11 : f32
              %91 = arith.subf %90, %cst_10 : f32
              %92 = arith.mulf %91, %87 : f32
              %93 = arith.mulf %92, %87 : f32
              %94 = arith.addf %93, %cst_12 : f32
              %95 = arith.select %88, %94, %2 : f32
              %96 = arith.cmpf olt, %87, %cst_9 : f32
              %97 = arith.andi %96, %89 : i1
              %98 = arith.xori %97, %true : i1
              %99 = arith.andi %98, %89 : i1
              %100 = arith.mulf %87, %cst_8 : f32
              %101 = arith.addf %100, %cst_7 : f32
              %102 = arith.mulf %101, %87 : f32
              %103 = arith.subf %102, %cst_6 : f32
              %104 = arith.mulf %103, %87 : f32
              %105 = arith.addf %104, %cst_5 : f32
              %106 = arith.select %97, %105, %95 : f32
              %107 = arith.select %99, %cst_4, %106 : f32
              %108 = arith.mulf %82, %107 : f32
              %109 = arith.addf %extracted, %108 : f32
              %110 = affine.apply #map(%arg6, %arg2, %arg4)
              %inserted_14 = tensor.insert %109 into %arg11[%110] : tensor<?xf32>
              affine.yield %inserted_14 : tensor<?xf32>
            }
            affine.yield %71 : tensor<?xf32>
          }
          affine.yield %36 : tensor<?xf32>
        }
        affine.yield %21 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

