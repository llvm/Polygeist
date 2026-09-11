#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<?x64xf32>, %arg4: memref<?x64xf32>, %arg5: memref<?x64xf32>, %arg6: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -2.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = bufferization.to_tensor %arg5 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?x64xf32>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = llvm.mlir.undef : f32
    %4 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %3 into %4[] : tensor<f32>
    %5 = tensor.empty() : tensor<f32>
    %inserted_3 = tensor.insert %3 into %5[] : tensor<f32>
    %6 = tensor.empty() : tensor<f32>
    %inserted_4 = tensor.insert %3 into %6[] : tensor<f32>
    %7 = arith.negf %arg2 : f32
    %8 = math.exp %7 : f32
    %9 = arith.subf %cst_2, %8 : f32
    %10 = arith.mulf %9, %9 : f32
    %11 = arith.mulf %arg2, %cst_1 : f32
    %12 = arith.mulf %11, %8 : f32
    %13 = arith.addf %12, %cst_2 : f32
    %14 = math.exp %11 : f32
    %15 = arith.subf %13, %14 : f32
    %16 = arith.divf %10, %15 : f32
    %17 = arith.mulf %16, %8 : f32
    %18 = arith.subf %arg2, %cst_2 : f32
    %19 = arith.mulf %17, %18 : f32
    %20 = math.powf %cst_1, %7 : f32
    %21 = arith.mulf %arg2, %cst_0 : f32
    %22 = math.exp %21 : f32
    %23 = arith.negf %22 : f32
    %24 = arith.index_cast %arg0 : i32 to index
    %25:4 = affine.for %arg7 = 0 to %24 iter_args(%arg8 = %inserted, %arg9 = %inserted_3, %arg10 = %inserted_4, %arg11 = %0) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<?x64xf32>) {
      %inserted_5 = tensor.insert %cst into %arg9[] : tensor<f32>
      %inserted_6 = tensor.insert %cst into %arg8[] : tensor<f32>
      %inserted_7 = tensor.insert %cst into %arg10[] : tensor<f32>
      %27 = polygeist.submap(%inserted_6, %2) {map = #map} : (tensor<f32>, index) -> tensor<?xf32>
      %28 = polygeist.submap(%inserted_5, %2) {map = #map} : (tensor<f32>, index) -> tensor<?xf32>
      %29 = polygeist.submap(%inserted_7, %2) {map = #map} : (tensor<f32>, index) -> tensor<?xf32>
      %30 = polygeist.submap(%1, %arg7, %2) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?xf32>
      %31 = polygeist.submap(%1, %arg7, %2) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?xf32>
      %32 = polygeist.submap(%arg11, %arg7, %2) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?xf32>
      %33 = polygeist.submap(%arg11, %arg7, %2) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?xf32>
      %34:4 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2, #map2, #map2, #map2, #map2], iterator_types = ["reduction"], library_call = ""} ins(%30, %31, %32 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%33, %29, %27, %28 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %in_9: f32, %out: f32, %out_10: f32, %out_11: f32, %out_12: f32):
        %39 = arith.mulf %16, %in : f32
        %40 = arith.mulf %19, %out_10 : f32
        %41 = arith.addf %39, %40 : f32
        %42 = arith.mulf %20, %out_12 : f32
        %43 = arith.addf %41, %42 : f32
        %44 = arith.mulf %23, %out_11 : f32
        %45 = arith.addf %43, %44 : f32
        linalg.yield %45, %in_8, %out_12, %in_9 : f32, f32, f32, f32
      } -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      %35 = polygeist.submapInverse(%arg11, %34#0, %arg7, %2) {map = #map1} : (tensor<?x64xf32>, tensor<?xf32>, index, index) -> tensor<?x64xf32>
      %36 = polygeist.submapInverse(%inserted_7, %34#1, %2) {map = #map} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
      %37 = polygeist.submapInverse(%inserted_5, %34#3, %2) {map = #map} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
      %38 = polygeist.submapInverse(%inserted_6, %34#2, %2) {map = #map} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
      affine.yield %38, %37, %36, %35 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<?x64xf32>
    }
    %26 = bufferization.to_memref %25#3 : memref<?x64xf32>
    memref.copy %26, %arg5 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

