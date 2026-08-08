#map = affine_map<(d0, d1, d2) -> (d2 + d0 * 56 + d1 * 8)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c5_i32 = arith.constant 5 : i32
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 7.000000e+00 : f32
    %cst_4 = arith.constant 4.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %0 = polygeist.submap(%arg1, %c2, %c7, %c8) {map = #map} : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf32>) {
    ^bb0(%out: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.muli %2, %c4_i32 : i32
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.addf %6, %cst_5 : f32
      %8 = arith.mulf %7, %cst_4 : f32
      %9 = arith.divf %8, %cst_3 : f32
      %10 = arith.subf %9, %cst_5 : f32
      %11 = arith.cmpf olt, %10, %cst_2 : f32
      %12 = arith.select %11, %cst_2, %10 : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.addi %3, %13 : i32
      %15 = arith.muli %14, %c5_i32 : i32
      %16 = arith.sitofp %13 : i32 to f32
      %17 = arith.subf %12, %16 : f32
      %18 = arith.subf %cst_1, %17 : f32
      %19 = arith.addi %13, %c1_i32 : i32
      %20 = arith.cmpi slt, %19, %c4_i32 : i32
      %21 = arith.select %20, %19, %13 : i32
      %22 = arith.addi %3, %21 : i32
      %23 = arith.muli %22, %c5_i32 : i32
      %24 = linalg.index 2 : index
      %25 = arith.index_cast %24 : index to i32
      %26 = arith.sitofp %25 : i32 to f32
      %27 = arith.addf %26, %cst_5 : f32
      %28 = arith.mulf %27, %cst_0 : f32
      %29 = arith.divf %28, %cst : f32
      %30 = arith.subf %29, %cst_5 : f32
      %31 = arith.cmpf olt, %30, %cst_2 : f32
      %32 = arith.select %31, %cst_2, %30 : f32
      %33 = arith.fptosi %32 : f32 to i32
      %34 = arith.addi %15, %33 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = memref.load %arg0[%35] : memref<?xf32>
      %37 = arith.mulf %36, %18 : f32
      %38 = arith.sitofp %33 : i32 to f32
      %39 = arith.subf %32, %38 : f32
      %40 = arith.subf %cst_1, %39 : f32
      %41 = arith.mulf %37, %40 : f32
      %42 = arith.addi %33, %c1_i32 : i32
      %43 = arith.cmpi slt, %42, %c5_i32 : i32
      %44 = arith.select %43, %42, %33 : i32
      %45 = arith.addi %15, %44 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = memref.load %arg0[%46] : memref<?xf32>
      %48 = arith.mulf %47, %18 : f32
      %49 = arith.mulf %48, %39 : f32
      %50 = arith.addf %41, %49 : f32
      %51 = arith.addi %23, %33 : i32
      %52 = arith.index_cast %51 : i32 to index
      %53 = memref.load %arg0[%52] : memref<?xf32>
      %54 = arith.mulf %53, %17 : f32
      %55 = arith.mulf %54, %40 : f32
      %56 = arith.addf %50, %55 : f32
      %57 = arith.addi %23, %44 : i32
      %58 = arith.index_cast %57 : i32 to index
      %59 = memref.load %arg0[%58] : memref<?xf32>
      %60 = arith.mulf %59, %17 : f32
      %61 = arith.mulf %60, %39 : f32
      %62 = arith.addf %56, %61 : f32
      linalg.yield %62 : f32
    }
    return
  }
}

