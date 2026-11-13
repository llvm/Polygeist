#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_cpu(%arg0: memref<?x3x8x8xf32>, %arg1: memref<?x6x6x2xf32>, %arg2: memref<?x3x6x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x6x6x2xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?x3x6x6xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0, 0, 0] [1, %c3, %c6, %c6] [1, 1, 1, 1] : tensor<?x3x6x6xf32> to tensor<?x?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %0[0, 0, 0, 0] [1, %c6, %c6, 1] [1, 1, 1, 1] : tensor<?x6x6x2xf32> to tensor<?x?xf32>
    %extracted_slice_4 = tensor.extract_slice %0[0, 0, 0, 1] [1, %c6, %c6, 1] [1, 1, 1, 1] : tensor<?x6x6x2xf32> to tensor<?x?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%extracted_slice_3, %extracted_slice_4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %4 = arith.addf %in, %cst : f32
      %5 = arith.mulf %4, %cst_0 : f32
      %6 = arith.mulf %5, %cst_2 : f32
      %7 = arith.addf %in_5, %cst : f32
      %8 = arith.mulf %7, %cst_0 : f32
      %9 = arith.mulf %8, %cst_2 : f32
      %10 = arith.fptosi %6 : f32 to i32
      %11 = arith.fptosi %9 : f32 to i32
      %12 = arith.addi %10, %c1_i32 : i32
      %13 = arith.addi %11, %c1_i32 : i32
      %14 = arith.sitofp %10 : i32 to f32
      %15 = arith.subf %6, %14 : f32
      %16 = arith.sitofp %11 : i32 to f32
      %17 = arith.subf %9, %16 : f32
      %18 = arith.cmpi sge, %10, %c0_i32 : i32
      %19 = arith.cmpi slt, %10, %c8_i32 : i32
      %20 = arith.cmpi sge, %11, %c0_i32 : i32
      %21 = arith.cmpi slt, %11, %c8_i32 : i32
      %22 = arith.andi %20, %21 : i1
      %23 = arith.andi %19, %22 : i1
      %24 = arith.andi %18, %23 : i1
      %25 = arith.cmpi sge, %12, %c0_i32 : i32
      %26 = arith.cmpi slt, %12, %c8_i32 : i32
      %27 = arith.andi %26, %22 : i1
      %28 = arith.andi %25, %27 : i1
      %29 = arith.cmpi sge, %13, %c0_i32 : i32
      %30 = arith.cmpi slt, %13, %c8_i32 : i32
      %31 = arith.andi %29, %30 : i1
      %32 = arith.andi %19, %31 : i1
      %33 = arith.andi %18, %32 : i1
      %34 = arith.andi %26, %31 : i1
      %35 = arith.andi %25, %34 : i1
      %36 = arith.subf %cst, %15 : f32
      %37 = arith.subf %cst, %17 : f32
      %38 = arith.mulf %36, %37 : f32
      %39 = arith.index_cast %11 : i32 to index
      %40 = arith.index_cast %10 : i32 to index
      %41 = arith.mulf %15, %37 : f32
      %42 = arith.index_cast %12 : i32 to index
      %43 = arith.mulf %36, %17 : f32
      %44 = arith.index_cast %13 : i32 to index
      %45 = arith.mulf %15, %17 : f32
      %46 = linalg.index 2 : index
      %47 = memref.load %arg0[%c0, %46, %39, %40] : memref<?x3x8x8xf32>
      %48 = arith.mulf %38, %47 : f32
      %49 = arith.addf %48, %cst_1 : f32
      %50 = arith.select %24, %49, %cst_1 : f32
      %51 = memref.load %arg0[%c0, %46, %39, %42] : memref<?x3x8x8xf32>
      %52 = arith.mulf %41, %51 : f32
      %53 = arith.addf %50, %52 : f32
      %54 = arith.select %28, %53, %50 : f32
      %55 = memref.load %arg0[%c0, %46, %44, %40] : memref<?x3x8x8xf32>
      %56 = arith.mulf %43, %55 : f32
      %57 = arith.addf %54, %56 : f32
      %58 = arith.select %33, %57, %54 : f32
      %59 = memref.load %arg0[%c0, %46, %44, %42] : memref<?x3x8x8xf32>
      %60 = arith.mulf %45, %59 : f32
      %61 = arith.addf %58, %60 : f32
      %62 = arith.select %35, %61, %58 : f32
      linalg.yield %62 : f32
    } -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %2 into %1[0, 0, 0, 0] [1, %c3, %c6, %c6] [1, 1, 1, 1] : tensor<?x?x?xf32> into tensor<?x3x6x6xf32>
    %3 = bufferization.to_memref %inserted_slice : memref<?x3x6x6xf32>
    memref.copy %3, %arg2 : memref<?x3x6x6xf32> to memref<?x3x6x6xf32>
    return
  }
}

