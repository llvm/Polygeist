#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant 5.000000e-01 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_5 : f32
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 7 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.addf %3, %cst_4 : f32
        %5 = arith.mulf %4, %cst_3 : f32
        %6 = arith.divf %5, %cst_2 : f32
        %7 = arith.subf %6, %cst_4 : f32
        %8 = arith.cmpf olt, %7, %cst_5 : f32
        %9 = arith.select %8, %cst_5, %7 : f32
        %10 = arith.fptosi %9 : f32 to i32
        %11 = arith.addi %1, %10 : i32
        %12 = arith.muli %11, %c5_i32 : i32
        %13 = arith.sitofp %10 : i32 to f32
        %14 = arith.subf %9, %13 : f32
        %15 = arith.subf %cst_1, %14 : f32
        %16 = arith.addi %10, %c1_i32 : i32
        %17 = arith.cmpi slt, %16, %c4_i32 : i32
        %18 = arith.select %17, %16, %10 : i32
        %19 = arith.addi %1, %18 : i32
        %20 = arith.muli %19, %c5_i32 : i32
        affine.for %arg4 = 0 to 8 {
          %21 = arith.index_cast %arg4 : index to i32
          %22 = arith.sitofp %21 : i32 to f32
          %23 = arith.addf %22, %cst_4 : f32
          %24 = arith.mulf %23, %cst_0 : f32
          %25 = arith.divf %24, %cst : f32
          %26 = arith.subf %25, %cst_4 : f32
          %27 = arith.cmpf olt, %26, %cst_5 : f32
          %28 = arith.select %27, %cst_5, %26 : f32
          %29 = arith.fptosi %28 : f32 to i32
          %30 = arith.addi %12, %29 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = affine.load %arg0[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
          %33 = arith.mulf %32, %15 : f32
          %34 = arith.sitofp %29 : i32 to f32
          %35 = arith.subf %28, %34 : f32
          %36 = arith.subf %cst_1, %35 : f32
          %37 = arith.mulf %33, %36 : f32
          %38 = memref.load %arg1[%31] : memref<?xf32>
          %39 = arith.addf %38, %37 : f32
          memref.store %39, %arg1[%31] : memref<?xf32>
          %40 = arith.addi %29, %c1_i32 : i32
          %41 = arith.cmpi slt, %40, %c5_i32 : i32
          %42 = arith.select %41, %40, %29 : i32
          %43 = arith.addi %12, %42 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = affine.load %arg0[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
          %46 = arith.mulf %45, %15 : f32
          %47 = arith.mulf %46, %35 : f32
          %48 = memref.load %arg1[%44] : memref<?xf32>
          %49 = arith.addf %48, %47 : f32
          memref.store %49, %arg1[%44] : memref<?xf32>
          %50 = arith.addi %20, %29 : i32
          %51 = arith.index_cast %50 : i32 to index
          %52 = affine.load %arg0[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
          %53 = arith.mulf %52, %14 : f32
          %54 = arith.mulf %53, %36 : f32
          %55 = memref.load %arg1[%51] : memref<?xf32>
          %56 = arith.addf %55, %54 : f32
          memref.store %56, %arg1[%51] : memref<?xf32>
          %57 = arith.addi %20, %42 : i32
          %58 = arith.index_cast %57 : i32 to index
          %59 = affine.load %arg0[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
          %60 = arith.mulf %59, %14 : f32
          %61 = arith.mulf %60, %35 : f32
          %62 = memref.load %arg1[%58] : memref<?xf32>
          %63 = arith.addf %62, %61 : f32
          memref.store %63, %arg1[%58] : memref<?xf32>
        }
      }
    }
    return
  }
}

