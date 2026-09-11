#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_durbin(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = tensor.empty() : tensor<f64>
    %3 = llvm.mlir.undef : f64
    %inserted = tensor.insert %3 into %2[] : tensor<f64>
    %4 = tensor.empty() : tensor<f64>
    %inserted_1 = tensor.insert %3 into %4[] : tensor<f64>
    %5 = tensor.empty() : tensor<f64>
    %inserted_2 = tensor.insert %3 into %5[] : tensor<f64>
    %6 = tensor.empty() : tensor<40xf64>
    %extracted = tensor.extract %1[%c0] : tensor<?xf64>
    %7 = arith.negf %extracted : f64
    %inserted_3 = tensor.insert %7 into %0[%c0] : tensor<?xf64>
    %inserted_4 = tensor.insert %cst into %inserted_1[] : tensor<f64>
    %extracted_5 = tensor.extract %1[%c0] : tensor<?xf64>
    %8 = arith.negf %extracted_5 : f64
    %inserted_6 = tensor.insert %8 into %inserted_2[] : tensor<f64>
    %9 = arith.index_cast %arg0 : i32 to index
    %10:5 = affine.for %arg3 = 1 to %9 iter_args(%arg4 = %inserted, %arg5 = %inserted_4, %arg6 = %inserted_6, %arg7 = %6, %arg8 = %inserted_3) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<40xf64>, tensor<?xf64>) {
      %extracted_7 = tensor.extract %arg6[] : tensor<f64>
      %12 = arith.mulf %extracted_7, %extracted_7 : f64
      %13 = arith.subf %cst, %12 : f64
      %extracted_8 = tensor.extract %arg5[] : tensor<f64>
      %14 = arith.mulf %13, %extracted_8 : f64
      %inserted_9 = tensor.insert %14 into %arg5[] : tensor<f64>
      %inserted_10 = tensor.insert %cst_0 into %arg4[] : tensor<f64>
      %15 = arith.subi %9, %c1 : index
      %16 = polygeist.submap(%inserted_10, %15) {map = #map} : (tensor<f64>, index) -> tensor<?xf64>
      %17 = polygeist.submap(%1, %arg3, %15) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %18 = polygeist.submap(%arg8, %15) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %19 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction"], library_call = ""} ins(%17, %18 : tensor<?xf64>, tensor<?xf64>) outs(%16 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_15: f64, %out: f64):
        %35 = arith.mulf %in, %in_15 : f64
        %36 = arith.addf %out, %35 : f64
        %37 = linalg.index 0 : index
        %38 = arith.cmpi slt, %37, %arg3 : index
        %39 = arith.select %38, %36, %out : f64
        linalg.yield %39 : f64
      } -> tensor<?xf64>
      %20 = polygeist.submapInverse(%inserted_10, %19, %15) {map = #map} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
      %extracted_11 = tensor.extract %1[%arg3] : tensor<?xf64>
      %extracted_12 = tensor.extract %20[] : tensor<f64>
      %21 = arith.addf %extracted_11, %extracted_12 : f64
      %22 = arith.negf %21 : f64
      %23 = arith.divf %22, %14 : f64
      %inserted_13 = tensor.insert %23 into %arg6[] : tensor<f64>
      %24 = arith.subi %9, %c1 : index
      %25 = polygeist.submap(%arg7, %24) {map = #map2} : (tensor<40xf64>, index) -> tensor<?xf64>
      %26 = polygeist.submap(%arg8, %24) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %27 = polygeist.submap(%arg8, %arg3, %24) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %28 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%26, %27 : tensor<?xf64>, tensor<?xf64>) outs(%25 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_15: f64, %out: f64):
        %35 = arith.mulf %23, %in_15 : f64
        %36 = arith.addf %in, %35 : f64
        %37 = linalg.index 0 : index
        %38 = arith.cmpi slt, %37, %arg3 : index
        %39 = arith.select %38, %36, %out : f64
        linalg.yield %39 : f64
      } -> tensor<?xf64>
      %29 = polygeist.submapInverse(%arg7, %28, %24) {map = #map2} : (tensor<40xf64>, tensor<?xf64>, index) -> tensor<40xf64>
      %30 = arith.subi %9, %c1 : index
      %31 = polygeist.submap(%29, %30) {map = #map2} : (tensor<40xf64>, index) -> tensor<?xf64>
      %32 = polygeist.submap(%arg8, %30) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %33 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%31 : tensor<?xf64>) outs(%32 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %35 = linalg.index 0 : index
        %36 = arith.cmpi slt, %35, %arg3 : index
        %37 = arith.select %36, %in, %out : f64
        linalg.yield %37 : f64
      } -> tensor<?xf64>
      %34 = polygeist.submapInverse(%arg8, %33, %30) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
      %inserted_14 = tensor.insert %23 into %34[%arg3] : tensor<?xf64>
      affine.yield %20, %inserted_9, %inserted_13, %29, %inserted_14 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<40xf64>, tensor<?xf64>
    }
    %11 = bufferization.to_memref %10#4 : memref<?xf64>
    memref.copy %11, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

