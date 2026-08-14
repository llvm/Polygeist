#map = affine_map<(d0, d1, d2) -> (d2 + d0 * 56 + d1 * 8)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant 7.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 5.000000e+00 : f32
    %cst_5 = arith.constant 8.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = polygeist.submap(%0, %c2, %c7, %c8) {map = #map} : (tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%1 : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.muli %6, %c4_i32 : i32
      %8 = linalg.index 1 : index
      %9 = arith.index_cast %8 : index to i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.addf %10, %cst : f32
      %12 = arith.mulf %11, %cst_0 : f32
      %13 = arith.divf %12, %cst_1 : f32
      %14 = arith.subf %13, %cst : f32
      %15 = arith.cmpf olt, %14, %cst_2 : f32
      %16 = arith.select %15, %cst_2, %14 : f32
      %17 = arith.fptosi %16 : f32 to i32
      %18 = arith.addi %7, %17 : i32
      %19 = arith.muli %18, %c5_i32 : i32
      %20 = arith.sitofp %17 : i32 to f32
      %21 = arith.subf %16, %20 : f32
      %22 = arith.subf %cst_3, %21 : f32
      %23 = arith.addi %17, %c1_i32 : i32
      %24 = arith.cmpi slt, %23, %c4_i32 : i32
      %25 = arith.select %24, %23, %17 : i32
      %26 = arith.addi %7, %25 : i32
      %27 = arith.muli %26, %c5_i32 : i32
      %28 = linalg.index 2 : index
      %29 = arith.index_cast %28 : index to i32
      %30 = arith.sitofp %29 : i32 to f32
      %31 = arith.addf %30, %cst : f32
      %32 = arith.mulf %31, %cst_4 : f32
      %33 = arith.divf %32, %cst_5 : f32
      %34 = arith.subf %33, %cst : f32
      %35 = arith.cmpf olt, %34, %cst_2 : f32
      %36 = arith.select %35, %cst_2, %34 : f32
      %37 = arith.fptosi %36 : f32 to i32
      %38 = arith.addi %19, %37 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = memref.load %arg0[%39] : memref<?xf32>
      %41 = arith.mulf %40, %22 : f32
      %42 = arith.sitofp %37 : i32 to f32
      %43 = arith.subf %36, %42 : f32
      %44 = arith.subf %cst_3, %43 : f32
      %45 = arith.mulf %41, %44 : f32
      %46 = arith.addi %37, %c1_i32 : i32
      %47 = arith.cmpi slt, %46, %c5_i32 : i32
      %48 = arith.select %47, %46, %37 : i32
      %49 = arith.addi %19, %48 : i32
      %50 = arith.index_cast %49 : i32 to index
      %51 = memref.load %arg0[%50] : memref<?xf32>
      %52 = arith.mulf %51, %22 : f32
      %53 = arith.mulf %52, %43 : f32
      %54 = arith.addf %45, %53 : f32
      %55 = arith.addi %27, %37 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = memref.load %arg0[%56] : memref<?xf32>
      %58 = arith.mulf %57, %21 : f32
      %59 = arith.mulf %58, %44 : f32
      %60 = arith.addf %54, %59 : f32
      %61 = arith.addi %27, %48 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = memref.load %arg0[%62] : memref<?xf32>
      %64 = arith.mulf %63, %21 : f32
      %65 = arith.mulf %64, %43 : f32
      %66 = arith.addf %60, %65 : f32
      linalg.yield %66 : f32
    } -> tensor<?x?x?xf32>
    %3 = polygeist.submapInverse(%0, %2, %c2, %c7, %c8) {map = #map} : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

