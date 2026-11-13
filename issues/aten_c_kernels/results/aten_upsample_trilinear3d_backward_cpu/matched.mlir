#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 504 + d1 + d2 * 72 + d3 * 9)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_trilinear3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 5.000000e+00 : f32
    %cst_5 = arith.constant 8.000000e+00 : f32
    %cst_6 = arith.constant 6.000000e+00 : f32
    %cst_7 = arith.constant 9.000000e+00 : f32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.addf %9, %cst_0 : f32
        %11 = arith.mulf %10, %cst_1 : f32
        %12 = arith.divf %11, %cst_2 : f32
        %13 = arith.subf %12, %cst_0 : f32
        %14 = arith.cmpf olt, %13, %cst : f32
        %15 = arith.select %14, %cst, %13 : f32
        %16 = arith.fptosi %15 : f32 to i32
        %17 = arith.addi %6, %16 : i32
        %18 = arith.muli %17, %c5_i32 : i32
        %19 = arith.sitofp %16 : i32 to f32
        %20 = arith.subf %15, %19 : f32
        %21 = arith.subf %cst_3, %20 : f32
        %22 = arith.addi %16, %c1_i32 : i32
        %23 = arith.cmpi slt, %22, %c4_i32 : i32
        %24 = arith.select %23, %22, %16 : i32
        %25 = arith.addi %6, %24 : i32
        %26 = arith.muli %25, %c5_i32 : i32
        %27 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %28 = arith.index_cast %arg6 : index to i32
          %29 = arith.sitofp %28 : i32 to f32
          %30 = arith.addf %29, %cst_0 : f32
          %31 = arith.mulf %30, %cst_4 : f32
          %32 = arith.divf %31, %cst_5 : f32
          %33 = arith.subf %32, %cst_0 : f32
          %34 = arith.cmpf olt, %33, %cst : f32
          %35 = arith.select %34, %cst, %33 : f32
          %36 = arith.fptosi %35 : f32 to i32
          %37 = arith.addi %18, %36 : i32
          %38 = arith.muli %37, %c6_i32 : i32
          %39 = arith.sitofp %36 : i32 to f32
          %40 = arith.subf %35, %39 : f32
          %41 = arith.subf %cst_3, %40 : f32
          %42 = arith.addi %36, %c1_i32 : i32
          %43 = arith.cmpi slt, %42, %c5_i32 : i32
          %44 = arith.select %43, %42, %36 : i32
          %45 = arith.addi %18, %44 : i32
          %46 = arith.muli %45, %c6_i32 : i32
          %47 = arith.addi %26, %36 : i32
          %48 = arith.muli %47, %c6_i32 : i32
          %49 = arith.addi %26, %44 : i32
          %50 = arith.muli %49, %c6_i32 : i32
          %51 = affine.for %arg8 = 0 to 9 iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
            %52 = arith.index_cast %arg8 : index to i32
            %53 = arith.sitofp %52 : i32 to f32
            %54 = arith.addf %53, %cst_0 : f32
            %55 = arith.mulf %54, %cst_6 : f32
            %56 = arith.divf %55, %cst_7 : f32
            %57 = arith.subf %56, %cst_0 : f32
            %58 = arith.cmpf olt, %57, %cst : f32
            %59 = arith.select %58, %cst, %57 : f32
            %60 = arith.fptosi %59 : f32 to i32
            %61 = arith.addi %38, %60 : i32
            %62 = arith.index_cast %61 : i32 to index
            %63 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted = tensor.extract %1[%63] : tensor<?xf32>
            %64 = arith.mulf %extracted, %21 : f32
            %65 = arith.mulf %64, %41 : f32
            %66 = arith.sitofp %60 : i32 to f32
            %67 = arith.subf %59, %66 : f32
            %68 = arith.subf %cst_3, %67 : f32
            %69 = arith.mulf %65, %68 : f32
            %extracted_8 = tensor.extract %arg9[%62] : tensor<?xf32>
            %70 = arith.addf %extracted_8, %69 : f32
            %inserted = tensor.insert %70 into %arg9[%62] : tensor<?xf32>
            %71 = arith.addi %60, %c1_i32 : i32
            %72 = arith.cmpi slt, %71, %c6_i32 : i32
            %73 = arith.select %72, %71, %60 : i32
            %74 = arith.addi %38, %73 : i32
            %75 = arith.index_cast %74 : i32 to index
            %76 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_9 = tensor.extract %1[%76] : tensor<?xf32>
            %77 = arith.mulf %extracted_9, %21 : f32
            %78 = arith.mulf %77, %41 : f32
            %79 = arith.mulf %78, %67 : f32
            %extracted_10 = tensor.extract %inserted[%75] : tensor<?xf32>
            %80 = arith.addf %extracted_10, %79 : f32
            %inserted_11 = tensor.insert %80 into %inserted[%75] : tensor<?xf32>
            %81 = arith.addi %46, %60 : i32
            %82 = arith.index_cast %81 : i32 to index
            %83 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_12 = tensor.extract %1[%83] : tensor<?xf32>
            %84 = arith.mulf %extracted_12, %21 : f32
            %85 = arith.mulf %84, %40 : f32
            %86 = arith.mulf %85, %68 : f32
            %extracted_13 = tensor.extract %inserted_11[%82] : tensor<?xf32>
            %87 = arith.addf %extracted_13, %86 : f32
            %inserted_14 = tensor.insert %87 into %inserted_11[%82] : tensor<?xf32>
            %88 = arith.addi %46, %73 : i32
            %89 = arith.index_cast %88 : i32 to index
            %90 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_15 = tensor.extract %1[%90] : tensor<?xf32>
            %91 = arith.mulf %extracted_15, %21 : f32
            %92 = arith.mulf %91, %40 : f32
            %93 = arith.mulf %92, %67 : f32
            %extracted_16 = tensor.extract %inserted_14[%89] : tensor<?xf32>
            %94 = arith.addf %extracted_16, %93 : f32
            %inserted_17 = tensor.insert %94 into %inserted_14[%89] : tensor<?xf32>
            %95 = arith.addi %48, %60 : i32
            %96 = arith.index_cast %95 : i32 to index
            %97 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_18 = tensor.extract %1[%97] : tensor<?xf32>
            %98 = arith.mulf %extracted_18, %20 : f32
            %99 = arith.mulf %98, %41 : f32
            %100 = arith.mulf %99, %68 : f32
            %extracted_19 = tensor.extract %inserted_17[%96] : tensor<?xf32>
            %101 = arith.addf %extracted_19, %100 : f32
            %inserted_20 = tensor.insert %101 into %inserted_17[%96] : tensor<?xf32>
            %102 = arith.addi %48, %73 : i32
            %103 = arith.index_cast %102 : i32 to index
            %104 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_21 = tensor.extract %1[%104] : tensor<?xf32>
            %105 = arith.mulf %extracted_21, %20 : f32
            %106 = arith.mulf %105, %41 : f32
            %107 = arith.mulf %106, %67 : f32
            %extracted_22 = tensor.extract %inserted_20[%103] : tensor<?xf32>
            %108 = arith.addf %extracted_22, %107 : f32
            %inserted_23 = tensor.insert %108 into %inserted_20[%103] : tensor<?xf32>
            %109 = arith.addi %50, %60 : i32
            %110 = arith.index_cast %109 : i32 to index
            %111 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_24 = tensor.extract %1[%111] : tensor<?xf32>
            %112 = arith.mulf %extracted_24, %20 : f32
            %113 = arith.mulf %112, %40 : f32
            %114 = arith.mulf %113, %68 : f32
            %extracted_25 = tensor.extract %inserted_23[%110] : tensor<?xf32>
            %115 = arith.addf %extracted_25, %114 : f32
            %inserted_26 = tensor.insert %115 into %inserted_23[%110] : tensor<?xf32>
            %116 = arith.addi %50, %73 : i32
            %117 = arith.index_cast %116 : i32 to index
            %118 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted_27 = tensor.extract %1[%118] : tensor<?xf32>
            %119 = arith.mulf %extracted_27, %20 : f32
            %120 = arith.mulf %119, %40 : f32
            %121 = arith.mulf %120, %67 : f32
            %extracted_28 = tensor.extract %inserted_26[%117] : tensor<?xf32>
            %122 = arith.addf %extracted_28, %121 : f32
            %inserted_29 = tensor.insert %122 into %inserted_26[%117] : tensor<?xf32>
            affine.yield %inserted_29 : tensor<?xf32>
          }
          affine.yield %51 : tensor<?xf32>
        }
        affine.yield %27 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

