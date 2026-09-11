#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_trilinear3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.000000e+00 : f32
    %cst_0 = arith.constant 6.000000e+00 : f32
    %cst_1 = arith.constant 8.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant 7.000000e+00 : f32
    %cst_5 = arith.constant 4.000000e+00 : f32
    %cst_6 = arith.constant 5.000000e-01 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_7 : f32
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 7 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.addf %3, %cst_6 : f32
        %5 = arith.mulf %4, %cst_5 : f32
        %6 = arith.divf %5, %cst_4 : f32
        %7 = arith.subf %6, %cst_6 : f32
        %8 = arith.cmpf olt, %7, %cst_7 : f32
        %9 = arith.select %8, %cst_7, %7 : f32
        %10 = arith.fptosi %9 : f32 to i32
        %11 = arith.addi %1, %10 : i32
        %12 = arith.muli %11, %c5_i32 : i32
        %13 = arith.sitofp %10 : i32 to f32
        %14 = arith.subf %9, %13 : f32
        %15 = arith.subf %cst_3, %14 : f32
        %16 = arith.addi %10, %c1_i32 : i32
        %17 = arith.cmpi slt, %16, %c4_i32 : i32
        %18 = arith.select %17, %16, %10 : i32
        %19 = arith.addi %1, %18 : i32
        %20 = arith.muli %19, %c5_i32 : i32
        affine.for %arg4 = 0 to 8 {
          %21 = arith.index_cast %arg4 : index to i32
          %22 = arith.sitofp %21 : i32 to f32
          %23 = arith.addf %22, %cst_6 : f32
          %24 = arith.mulf %23, %cst_2 : f32
          %25 = arith.divf %24, %cst_1 : f32
          %26 = arith.subf %25, %cst_6 : f32
          %27 = arith.cmpf olt, %26, %cst_7 : f32
          %28 = arith.select %27, %cst_7, %26 : f32
          %29 = arith.fptosi %28 : f32 to i32
          %30 = arith.addi %12, %29 : i32
          %31 = arith.muli %30, %c6_i32 : i32
          %32 = arith.sitofp %29 : i32 to f32
          %33 = arith.subf %28, %32 : f32
          %34 = arith.subf %cst_3, %33 : f32
          %35 = arith.addi %29, %c1_i32 : i32
          %36 = arith.cmpi slt, %35, %c5_i32 : i32
          %37 = arith.select %36, %35, %29 : i32
          %38 = arith.addi %12, %37 : i32
          %39 = arith.muli %38, %c6_i32 : i32
          %40 = arith.addi %20, %29 : i32
          %41 = arith.muli %40, %c6_i32 : i32
          %42 = arith.addi %20, %37 : i32
          %43 = arith.muli %42, %c6_i32 : i32
          affine.for %arg5 = 0 to 9 {
            %44 = arith.index_cast %arg5 : index to i32
            %45 = arith.sitofp %44 : i32 to f32
            %46 = arith.addf %45, %cst_6 : f32
            %47 = arith.mulf %46, %cst_0 : f32
            %48 = arith.divf %47, %cst : f32
            %49 = arith.subf %48, %cst_6 : f32
            %50 = arith.cmpf olt, %49, %cst_7 : f32
            %51 = arith.select %50, %cst_7, %49 : f32
            %52 = arith.fptosi %51 : f32 to i32
            %53 = arith.addi %31, %52 : i32
            %54 = arith.index_cast %53 : i32 to index
            %55 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %56 = arith.mulf %55, %15 : f32
            %57 = arith.mulf %56, %34 : f32
            %58 = arith.sitofp %52 : i32 to f32
            %59 = arith.subf %51, %58 : f32
            %60 = arith.subf %cst_3, %59 : f32
            %61 = arith.mulf %57, %60 : f32
            %62 = memref.load %arg1[%54] : memref<?xf32>
            %63 = arith.addf %62, %61 : f32
            memref.store %63, %arg1[%54] : memref<?xf32>
            %64 = arith.addi %52, %c1_i32 : i32
            %65 = arith.cmpi slt, %64, %c6_i32 : i32
            %66 = arith.select %65, %64, %52 : i32
            %67 = arith.addi %31, %66 : i32
            %68 = arith.index_cast %67 : i32 to index
            %69 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %70 = arith.mulf %69, %15 : f32
            %71 = arith.mulf %70, %34 : f32
            %72 = arith.mulf %71, %59 : f32
            %73 = memref.load %arg1[%68] : memref<?xf32>
            %74 = arith.addf %73, %72 : f32
            memref.store %74, %arg1[%68] : memref<?xf32>
            %75 = arith.addi %39, %52 : i32
            %76 = arith.index_cast %75 : i32 to index
            %77 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %78 = arith.mulf %77, %15 : f32
            %79 = arith.mulf %78, %33 : f32
            %80 = arith.mulf %79, %60 : f32
            %81 = memref.load %arg1[%76] : memref<?xf32>
            %82 = arith.addf %81, %80 : f32
            memref.store %82, %arg1[%76] : memref<?xf32>
            %83 = arith.addi %39, %66 : i32
            %84 = arith.index_cast %83 : i32 to index
            %85 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %86 = arith.mulf %85, %15 : f32
            %87 = arith.mulf %86, %33 : f32
            %88 = arith.mulf %87, %59 : f32
            %89 = memref.load %arg1[%84] : memref<?xf32>
            %90 = arith.addf %89, %88 : f32
            memref.store %90, %arg1[%84] : memref<?xf32>
            %91 = arith.addi %41, %52 : i32
            %92 = arith.index_cast %91 : i32 to index
            %93 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %94 = arith.mulf %93, %14 : f32
            %95 = arith.mulf %94, %34 : f32
            %96 = arith.mulf %95, %60 : f32
            %97 = memref.load %arg1[%92] : memref<?xf32>
            %98 = arith.addf %97, %96 : f32
            memref.store %98, %arg1[%92] : memref<?xf32>
            %99 = arith.addi %41, %66 : i32
            %100 = arith.index_cast %99 : i32 to index
            %101 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %102 = arith.mulf %101, %14 : f32
            %103 = arith.mulf %102, %34 : f32
            %104 = arith.mulf %103, %59 : f32
            %105 = memref.load %arg1[%100] : memref<?xf32>
            %106 = arith.addf %105, %104 : f32
            memref.store %106, %arg1[%100] : memref<?xf32>
            %107 = arith.addi %43, %52 : i32
            %108 = arith.index_cast %107 : i32 to index
            %109 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %110 = arith.mulf %109, %14 : f32
            %111 = arith.mulf %110, %33 : f32
            %112 = arith.mulf %111, %60 : f32
            %113 = memref.load %arg1[%108] : memref<?xf32>
            %114 = arith.addf %113, %112 : f32
            memref.store %114, %arg1[%108] : memref<?xf32>
            %115 = arith.addi %43, %66 : i32
            %116 = arith.index_cast %115 : i32 to index
            %117 = affine.load %arg0[%arg2 * 504 + %arg5 + %arg3 * 72 + %arg4 * 9] : memref<?xf32>
            %118 = arith.mulf %117, %14 : f32
            %119 = arith.mulf %118, %33 : f32
            %120 = arith.mulf %119, %59 : f32
            %121 = memref.load %arg1[%116] : memref<?xf32>
            %122 = arith.addf %121, %120 : f32
            memref.store %122, %arg1[%116] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}

