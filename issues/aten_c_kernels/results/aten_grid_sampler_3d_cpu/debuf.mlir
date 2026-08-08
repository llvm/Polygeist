#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4)[s0, s1] -> (0, d1, s0, s1, d0)>
#map2 = affine_map<(d0)[s0, s1] -> (0, s0, s1, d0, 0)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
#map4 = affine_map<(d0)[s0, s1] -> (0, s0, s1, d0, 1)>
#map5 = affine_map<(d0)[s0, s1] -> (0, s0, s1, d0, 2)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_3d_cpu(%arg0: memref<?x2x6x7x8xf32>, %arg1: memref<?x4x5x6x3xf32>, %arg2: memref<?x2x4x5x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c7_i32 = arith.constant 7 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant 5.000000e+00 : f32
    %cst_3 = arith.constant 6.000000e+00 : f32
    %cst_4 = arith.constant 7.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x4x5x6x3xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?x2x4x5x6xf32>
    %2 = affine.for %arg3 = 0 to 4 iter_args(%arg4 = %1) -> (tensor<?x2x4x5x6xf32>) {
      %4 = affine.for %arg5 = 0 to 5 iter_args(%arg6 = %arg4) -> (tensor<?x2x4x5x6xf32>) {
        %extracted_slice = tensor.extract_slice %arg6[0, 0, %arg3, %arg5, 0] [1, %c2, 1, 1, %c6] [1, 1, 1, 1, 1] : tensor<?x2x4x5x6xf32> to tensor<?x?xf32>
        %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_1 : f32
        } -> tensor<?x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg6[0, 0, %arg3, %arg5, 0] [1, %c2, 1, 1, %c6] [1, 1, 1, 1, 1] : tensor<?x?xf32> into tensor<?x2x4x5x6xf32>
        %6 = polygeist.submap(%inserted_slice, %arg3, %arg5, %c6, %c2, %c2, %c2, %c2) {map = #map1} : (tensor<?x2x4x5x6xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
        %7 = polygeist.submap(%0, %arg3, %arg5, %c6) {map = #map2} : (tensor<?x4x5x6x3xf32>, index, index, index) -> tensor<?xf32>
        %8 = polygeist.submap(%7, %c6, %c2, %c2, %c2, %c2) {map = #map3} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
        %9 = polygeist.submap(%0, %arg3, %arg5, %c6) {map = #map4} : (tensor<?x4x5x6x3xf32>, index, index, index) -> tensor<?xf32>
        %10 = polygeist.submap(%9, %c6, %c2, %c2, %c2, %c2) {map = #map3} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
        %11 = polygeist.submap(%0, %arg3, %arg5, %c6) {map = #map5} : (tensor<?x4x5x6x3xf32>, index, index, index) -> tensor<?xf32>
        %12 = polygeist.submap(%11, %c6, %c2, %c2, %c2, %c2) {map = #map3} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
        %13 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%8, %10, %12 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%6 : tensor<?x?x?x?x?xf32>) {
        ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
          %15 = arith.addf %in, %cst : f32
          %16 = arith.mulf %15, %cst_0 : f32
          %17 = arith.mulf %16, %cst_4 : f32
          %18 = arith.addf %in_5, %cst : f32
          %19 = arith.mulf %18, %cst_0 : f32
          %20 = arith.mulf %19, %cst_3 : f32
          %21 = arith.addf %in_6, %cst : f32
          %22 = arith.mulf %21, %cst_0 : f32
          %23 = arith.mulf %22, %cst_2 : f32
          %24 = arith.fptosi %17 : f32 to i32
          %25 = arith.fptosi %20 : f32 to i32
          %26 = arith.fptosi %23 : f32 to i32
          %27 = arith.sitofp %24 : i32 to f32
          %28 = arith.subf %17, %27 : f32
          %29 = arith.sitofp %25 : i32 to f32
          %30 = arith.subf %20, %29 : f32
          %31 = arith.sitofp %26 : i32 to f32
          %32 = arith.subf %23, %31 : f32
          %33 = arith.subf %cst, %28 : f32
          %34 = arith.subf %cst, %30 : f32
          %35 = arith.subf %cst, %32 : f32
          %36 = linalg.index 1 : index
          %37 = linalg.index 2 : index
          %38 = arith.index_cast %37 : index to i32
          %39 = arith.addi %26, %38 : i32
          %40 = arith.cmpi sge, %39, %c0_i32 : i32
          %41 = arith.cmpi slt, %39, %c6_i32 : i32
          %42 = arith.index_cast %39 : i32 to index
          %43 = arith.cmpi ne, %38, %c0_i32 : i32
          %44 = arith.select %43, %32, %35 : f32
          %45 = linalg.index 3 : index
          %46 = arith.index_cast %45 : index to i32
          %47 = arith.addi %25, %46 : i32
          %48 = arith.cmpi sge, %47, %c0_i32 : i32
          %49 = arith.cmpi slt, %47, %c7_i32 : i32
          %50 = arith.index_cast %47 : i32 to index
          %51 = arith.cmpi ne, %46, %c0_i32 : i32
          %52 = arith.select %51, %30, %34 : f32
          %53 = linalg.index 4 : index
          %54 = arith.index_cast %53 : index to i32
          %55 = arith.addi %24, %54 : i32
          %56 = arith.cmpi sge, %55, %c0_i32 : i32
          %57 = arith.cmpi slt, %55, %c8_i32 : i32
          %58 = arith.andi %56, %57 : i1
          %59 = arith.andi %49, %58 : i1
          %60 = arith.andi %48, %59 : i1
          %61 = arith.andi %41, %60 : i1
          %62 = arith.andi %40, %61 : i1
          %63 = arith.index_cast %55 : i32 to index
          %64 = memref.load %arg0[%c0, %36, %42, %50, %63] : memref<?x2x6x7x8xf32>
          %65 = arith.mulf %64, %44 : f32
          %66 = arith.mulf %65, %52 : f32
          %67 = arith.cmpi ne, %54, %c0_i32 : i32
          %68 = arith.select %67, %28, %33 : f32
          %69 = arith.mulf %66, %68 : f32
          %70 = arith.addf %out, %69 : f32
          %71 = arith.select %62, %70, %out : f32
          linalg.yield %71 : f32
        } -> tensor<?x?x?x?x?xf32>
        %14 = polygeist.submapInverse(%inserted_slice, %13, %arg3, %arg5, %c6, %c2, %c2, %c2, %c2) {map = #map1} : (tensor<?x2x4x5x6xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x2x4x5x6xf32>
        affine.yield %14 : tensor<?x2x4x5x6xf32>
      }
      affine.yield %4 : tensor<?x2x4x5x6xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?x2x4x5x6xf32>
    memref.copy %3, %arg2 : memref<?x2x4x5x6xf32> to memref<?x2x4x5x6xf32>
    return
  }
}

