#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_cpu(%arg0: memref<?x3x8x8xf32>, %arg1: memref<?x6x6x2xf32>, %arg2: memref<?x3x6x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 7.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %subview = memref.subview %arg1[0, 0, 0, 0] [1, %c6, %c6, 1] [1, 1, 1, 1] : memref<?x6x6x2xf32> to memref<?x?xf32, strided<[12, 2]>>
    %subview_3 = memref.subview %arg1[0, 0, 0, 1] [1, %c6, %c6, 1] [1, 1, 1, 1] : memref<?x6x6x2xf32> to memref<?x?xf32, strided<[12, 2], offset: 1>>
    %subview_4 = memref.subview %arg2[0, 0, 0, 0] [1, %c3, %c6, %c6] [1, 1, 1, 1] : memref<?x3x6x6xf32> to memref<?x?x?xf32, strided<[36, 6, 1]>>
    linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview, %subview_3 : memref<?x?xf32, strided<[12, 2]>>, memref<?x?xf32, strided<[12, 2], offset: 1>>) outs(%subview_4 : memref<?x?x?xf32, strided<[36, 6, 1]>>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %0 = arith.addf %in, %cst_2 : f32
      %1 = arith.mulf %0, %cst_1 : f32
      %2 = arith.mulf %1, %cst : f32
      %3 = arith.addf %in_5, %cst_2 : f32
      %4 = arith.mulf %3, %cst_1 : f32
      %5 = arith.mulf %4, %cst : f32
      %6 = arith.fptosi %2 : f32 to i32
      %7 = arith.fptosi %5 : f32 to i32
      %8 = arith.addi %6, %c1_i32 : i32
      %9 = arith.addi %7, %c1_i32 : i32
      %10 = arith.sitofp %6 : i32 to f32
      %11 = arith.subf %2, %10 : f32
      %12 = arith.sitofp %7 : i32 to f32
      %13 = arith.subf %5, %12 : f32
      %14 = arith.cmpi sge, %6, %c0_i32 : i32
      %15 = arith.cmpi slt, %6, %c8_i32 : i32
      %16 = arith.cmpi sge, %7, %c0_i32 : i32
      %17 = arith.cmpi slt, %7, %c8_i32 : i32
      %18 = arith.andi %16, %17 : i1
      %19 = arith.andi %15, %18 : i1
      %20 = arith.andi %14, %19 : i1
      %21 = arith.cmpi sge, %8, %c0_i32 : i32
      %22 = arith.cmpi slt, %8, %c8_i32 : i32
      %23 = arith.andi %22, %18 : i1
      %24 = arith.andi %21, %23 : i1
      %25 = arith.cmpi sge, %9, %c0_i32 : i32
      %26 = arith.cmpi slt, %9, %c8_i32 : i32
      %27 = arith.andi %25, %26 : i1
      %28 = arith.andi %15, %27 : i1
      %29 = arith.andi %14, %28 : i1
      %30 = arith.andi %22, %27 : i1
      %31 = arith.andi %21, %30 : i1
      %32 = arith.subf %cst_2, %11 : f32
      %33 = arith.subf %cst_2, %13 : f32
      %34 = arith.mulf %32, %33 : f32
      %35 = arith.index_cast %7 : i32 to index
      %36 = arith.index_cast %6 : i32 to index
      %37 = arith.mulf %11, %33 : f32
      %38 = arith.index_cast %8 : i32 to index
      %39 = arith.mulf %32, %13 : f32
      %40 = arith.index_cast %9 : i32 to index
      %41 = arith.mulf %11, %13 : f32
      %42 = linalg.index 2 : index
      %43 = memref.load %arg0[%c0, %42, %35, %36] : memref<?x3x8x8xf32>
      %44 = arith.mulf %34, %43 : f32
      %45 = arith.addf %44, %cst_0 : f32
      %46 = arith.select %20, %45, %cst_0 : f32
      %47 = memref.load %arg0[%c0, %42, %35, %38] : memref<?x3x8x8xf32>
      %48 = arith.mulf %37, %47 : f32
      %49 = arith.addf %46, %48 : f32
      %50 = arith.select %24, %49, %46 : f32
      %51 = memref.load %arg0[%c0, %42, %40, %36] : memref<?x3x8x8xf32>
      %52 = arith.mulf %39, %51 : f32
      %53 = arith.addf %50, %52 : f32
      %54 = arith.select %29, %53, %50 : f32
      %55 = memref.load %arg0[%c0, %42, %40, %38] : memref<?x3x8x8xf32>
      %56 = arith.mulf %41, %55 : f32
      %57 = arith.addf %54, %56 : f32
      %58 = arith.select %31, %57, %54 : f32
      linalg.yield %58 : f32
    }
    return
  }
}

