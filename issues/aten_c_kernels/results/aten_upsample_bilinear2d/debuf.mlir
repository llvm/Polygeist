#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d(%arg0: memref<?x3x4x4xf32>, %arg1: memref<?x3x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x3x8x8xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%0 : tensor<?x3x8x8xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = linalg.index 2 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.remsi %6, %c2_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst : f32
      %10 = arith.subf %cst_0, %9 : f32
      %11 = arith.divsi %6, %c2_i32 : i32
      %12 = arith.index_cast %11 : i32 to index
      %13 = arith.addi %11, %c1_i32 : i32
      %14 = arith.cmpi slt, %13, %c4_i32 : i32
      %15 = arith.select %14, %13, %11 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.cmpi slt, %5, %c0 : index
      %18 = arith.subi %c-1, %5 : index
      %19 = arith.select %17, %18, %5 : index
      %20 = arith.divsi %19, %c2 : index
      %21 = arith.subi %c-1, %20 : index
      %22 = arith.select %17, %21, %20 : index
      %23 = linalg.index 3 : index
      %24 = arith.index_cast %23 : index to i32
      %25 = arith.remsi %24, %c2_i32 : i32
      %26 = arith.sitofp %25 : i32 to f32
      %27 = arith.mulf %26, %cst : f32
      %28 = arith.subf %cst_0, %27 : f32
      %29 = arith.divsi %24, %c2_i32 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.cmpi slt, %23, %c0 : index
      %32 = arith.subi %c-1, %23 : index
      %33 = arith.select %31, %32, %23 : index
      %34 = arith.divsi %33, %c2 : index
      %35 = arith.subi %c-1, %34 : index
      %36 = arith.select %31, %35, %34 : index
      %37 = memref.load %arg0[%3, %4, %22, %36] : memref<?x3x4x4xf32>
      %38 = arith.mulf %28, %37 : f32
      %39 = arith.addi %29, %c1_i32 : i32
      %40 = arith.cmpi slt, %39, %c4_i32 : i32
      %41 = arith.select %40, %39, %29 : i32
      %42 = arith.index_cast %41 : i32 to index
      %43 = memref.load %arg0[%3, %4, %12, %42] : memref<?x3x4x4xf32>
      %44 = arith.mulf %27, %43 : f32
      %45 = arith.addf %38, %44 : f32
      %46 = arith.mulf %10, %45 : f32
      %47 = memref.load %arg0[%3, %4, %16, %30] : memref<?x3x4x4xf32>
      %48 = arith.mulf %28, %47 : f32
      %49 = memref.load %arg0[%3, %4, %16, %42] : memref<?x3x4x4xf32>
      %50 = arith.mulf %27, %49 : f32
      %51 = arith.addf %48, %50 : f32
      %52 = arith.mulf %9, %51 : f32
      %53 = arith.addf %46, %52 : f32
      linalg.yield %53 : f32
    } -> tensor<?x3x8x8xf32>
    %2 = bufferization.to_memref %1 : memref<?x3x8x8xf32>
    memref.copy %2, %arg1 : memref<?x3x8x8xf32> to memref<?x3x8x8xf32>
    return
  }
}

