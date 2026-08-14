#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4)[s0, s1] -> (0, d1, s0, s1, d0)>
#map2 = affine_map<(d0)[s0, s1] -> (0, s0, s1, d0, 0)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
#map4 = affine_map<(d0)[s0, s1] -> (0, s0, s1, d0, 1)>
#map5 = affine_map<(d0)[s0, s1] -> (0, s0, s1, d0, 2)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_3d_cpu(%arg0: memref<?x2x6x7x8xf32>, %arg1: memref<?x4x5x6x3xf32>, %arg2: memref<?x2x4x5x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c6 = arith.constant 6 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 6.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %cst_4 = arith.constant 1.000000e+00 : f32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 5 {
        %subview = memref.subview %arg2[0, 0, %arg3, %arg4, 0] [1, %c2, 1, 1, %c6] [1, 1, 1, 1, 1] : memref<?x2x4x5x6xf32> to memref<?x?xf32, strided<[120, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[120, 1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_2 : f32
        }
        %0 = polygeist.submap(%arg2, %arg3, %arg4, %c6, %c2, %c2, %c2, %c2) {map = #map1} : (memref<?x2x4x5x6xf32>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
        %1 = polygeist.submap(%arg1, %arg3, %arg4, %c6) {map = #map2} : (memref<?x4x5x6x3xf32>, index, index, index) -> memref<?xf32>
        %2 = polygeist.submap(%1, %c6, %c2, %c2, %c2, %c2) {map = #map3} : (memref<?xf32>, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
        %3 = polygeist.submap(%arg1, %arg3, %arg4, %c6) {map = #map4} : (memref<?x4x5x6x3xf32>, index, index, index) -> memref<?xf32>
        %4 = polygeist.submap(%3, %c6, %c2, %c2, %c2, %c2) {map = #map3} : (memref<?xf32>, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
        %5 = polygeist.submap(%arg1, %arg3, %arg4, %c6) {map = #map5} : (memref<?x4x5x6x3xf32>, index, index, index) -> memref<?xf32>
        %6 = polygeist.submap(%5, %c6, %c2, %c2, %c2, %c2) {map = #map3} : (memref<?xf32>, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
        linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%2, %4, %6 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%0 : memref<?x?x?x?x?xf32>) {
        ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
          %7 = arith.addf %in, %cst_4 : f32
          %8 = arith.mulf %7, %cst_3 : f32
          %9 = arith.mulf %8, %cst : f32
          %10 = arith.addf %in_5, %cst_4 : f32
          %11 = arith.mulf %10, %cst_3 : f32
          %12 = arith.mulf %11, %cst_0 : f32
          %13 = arith.addf %in_6, %cst_4 : f32
          %14 = arith.mulf %13, %cst_3 : f32
          %15 = arith.mulf %14, %cst_1 : f32
          %16 = arith.fptosi %9 : f32 to i32
          %17 = arith.fptosi %12 : f32 to i32
          %18 = arith.fptosi %15 : f32 to i32
          %19 = arith.sitofp %16 : i32 to f32
          %20 = arith.subf %9, %19 : f32
          %21 = arith.sitofp %17 : i32 to f32
          %22 = arith.subf %12, %21 : f32
          %23 = arith.sitofp %18 : i32 to f32
          %24 = arith.subf %15, %23 : f32
          %25 = arith.subf %cst_4, %20 : f32
          %26 = arith.subf %cst_4, %22 : f32
          %27 = arith.subf %cst_4, %24 : f32
          %28 = linalg.index 1 : index
          %29 = linalg.index 2 : index
          %30 = arith.index_cast %29 : index to i32
          %31 = arith.addi %18, %30 : i32
          %32 = arith.cmpi sge, %31, %c0_i32 : i32
          %33 = arith.cmpi slt, %31, %c6_i32 : i32
          %34 = arith.index_cast %31 : i32 to index
          %35 = arith.cmpi ne, %30, %c0_i32 : i32
          %36 = arith.select %35, %24, %27 : f32
          %37 = linalg.index 3 : index
          %38 = arith.index_cast %37 : index to i32
          %39 = arith.addi %17, %38 : i32
          %40 = arith.cmpi sge, %39, %c0_i32 : i32
          %41 = arith.cmpi slt, %39, %c7_i32 : i32
          %42 = arith.index_cast %39 : i32 to index
          %43 = arith.cmpi ne, %38, %c0_i32 : i32
          %44 = arith.select %43, %22, %26 : f32
          %45 = linalg.index 4 : index
          %46 = arith.index_cast %45 : index to i32
          %47 = arith.addi %16, %46 : i32
          %48 = arith.cmpi sge, %47, %c0_i32 : i32
          %49 = arith.cmpi slt, %47, %c8_i32 : i32
          %50 = arith.andi %48, %49 : i1
          %51 = arith.andi %41, %50 : i1
          %52 = arith.andi %40, %51 : i1
          %53 = arith.andi %33, %52 : i1
          %54 = arith.andi %32, %53 : i1
          %55 = arith.index_cast %47 : i32 to index
          %56 = memref.load %arg0[%c0, %28, %34, %42, %55] : memref<?x2x6x7x8xf32>
          %57 = arith.mulf %56, %36 : f32
          %58 = arith.mulf %57, %44 : f32
          %59 = arith.cmpi ne, %46, %c0_i32 : i32
          %60 = arith.select %59, %20, %25 : f32
          %61 = arith.mulf %58, %60 : f32
          %62 = arith.addf %out, %61 : f32
          %63 = arith.select %54, %62, %out : f32
          linalg.yield %63 : f32
        }
      }
    }
    return
  }
}

