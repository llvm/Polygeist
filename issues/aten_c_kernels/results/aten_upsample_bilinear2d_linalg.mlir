#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d(%arg0: memref<?x3x4x4xf32>, %arg1: memref<?x3x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c-1 = arith.constant -1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %subview = memref.subview %arg1[0, 0, 0, 0] [%c2, %c3, %c8, %c8] [1, 1, 1, 1] : memref<?x3x8x8xf32> to memref<?x?x?x?xf32, strided<[192, 64, 8, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?x?xf32, strided<[192, 64, 8, 1]>>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.remsi %3, %c2_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_0 : f32
      %7 = arith.subf %cst, %6 : f32
      %8 = arith.divsi %3, %c2_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.addi %8, %c1_i32 : i32
      %11 = arith.cmpi slt, %10, %c4_i32 : i32
      %12 = arith.select %11, %10, %8 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.cmpi slt, %2, %c0 : index
      %15 = arith.subi %c-1, %2 : index
      %16 = arith.select %14, %15, %2 : index
      %17 = arith.divsi %16, %c2 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = linalg.index 3 : index
      %21 = arith.index_cast %20 : index to i32
      %22 = arith.remsi %21, %c2_i32 : i32
      %23 = arith.sitofp %22 : i32 to f32
      %24 = arith.mulf %23, %cst_0 : f32
      %25 = arith.subf %cst, %24 : f32
      %26 = arith.divsi %21, %c2_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.cmpi slt, %20, %c0 : index
      %29 = arith.subi %c-1, %20 : index
      %30 = arith.select %28, %29, %20 : index
      %31 = arith.divsi %30, %c2 : index
      %32 = arith.subi %c-1, %31 : index
      %33 = arith.select %28, %32, %31 : index
      %34 = memref.load %arg0[%0, %1, %19, %33] : memref<?x3x4x4xf32>
      %35 = arith.mulf %25, %34 : f32
      %36 = arith.addi %26, %c1_i32 : i32
      %37 = arith.cmpi slt, %36, %c4_i32 : i32
      %38 = arith.select %37, %36, %26 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = memref.load %arg0[%0, %1, %9, %39] : memref<?x3x4x4xf32>
      %41 = arith.mulf %24, %40 : f32
      %42 = arith.addf %35, %41 : f32
      %43 = arith.mulf %7, %42 : f32
      %44 = memref.load %arg0[%0, %1, %13, %27] : memref<?x3x4x4xf32>
      %45 = arith.mulf %25, %44 : f32
      %46 = memref.load %arg0[%0, %1, %13, %39] : memref<?x3x4x4xf32>
      %47 = arith.mulf %24, %46 : f32
      %48 = arith.addf %45, %47 : f32
      %49 = arith.mulf %6, %48 : f32
      %50 = arith.addf %43, %49 : f32
      linalg.yield %50 : f32
    }
    return
  }
}

