#map = affine_map<(d0, d1) -> (d1 + d0 * 7)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_linear1d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant 7.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c7 = arith.constant 7 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = polygeist.submap(%0, %c2, %c7) {map = #map} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%1 : tensor<?x?xf32>) {
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
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %arg0[%19] : memref<?xf32>
      %21 = arith.sitofp %17 : i32 to f32
      %22 = arith.subf %16, %21 : f32
      %23 = arith.subf %cst_3, %22 : f32
      %24 = arith.mulf %20, %23 : f32
      %25 = arith.addi %17, %c1_i32 : i32
      %26 = arith.cmpi slt, %25, %c4_i32 : i32
      %27 = arith.select %26, %25, %17 : i32
      %28 = arith.addi %7, %27 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = memref.load %arg0[%29] : memref<?xf32>
      %31 = arith.mulf %30, %22 : f32
      %32 = arith.addf %24, %31 : f32
      linalg.yield %32 : f32
    } -> tensor<?x?xf32>
    %3 = polygeist.submapInverse(%0, %2, %c2, %c7) {map = #map} : (tensor<?xf32>, tensor<?x?xf32>, index, index) -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

