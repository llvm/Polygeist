#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0)[s0] -> (s0, d0)>
#map3 = affine_map<(d0)[s0] -> (d0, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map5 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map6 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_ludcmp(%arg0: i32, %arg1: memref<?x40xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg1 : memref<?x40xf64>
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = tensor.empty() : tensor<f64>
    %6 = llvm.mlir.undef : f64
    %inserted = tensor.insert %6 into %5[] : tensor<f64>
    %7:2 = affine.for %arg5 = 0 to %4 iter_args(%arg6 = %inserted, %arg7 = %3) -> (tensor<f64>, tensor<?x40xf64>) {
      %13:2 = affine.for %arg8 = 0 to #map(%arg5) iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<f64>, tensor<?x40xf64>) {
        %extracted = tensor.extract %arg10[%arg5, %arg8] : tensor<?x40xf64>
        %inserted_0 = tensor.insert %extracted into %arg9[] : tensor<f64>
        %15 = arith.subi %arg5, %c1 : index
        %16 = polygeist.submap(%inserted_0, %15) {map = #map1} : (tensor<f64>, index) -> tensor<?xf64>
        %17 = polygeist.submap(%arg10, %arg5, %15) {map = #map2} : (tensor<?x40xf64>, index, index) -> tensor<?xf64>
        %18 = polygeist.submap(%arg10, %arg8, %15) {map = #map3} : (tensor<?x40xf64>, index, index) -> tensor<?xf64>
        %19 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%17, %18 : tensor<?xf64>, tensor<?xf64>) outs(%16 : tensor<?xf64>) {
        ^bb0(%in: f64, %in_4: f64, %out: f64):
          %22 = arith.mulf %in, %in_4 : f64
          %23 = arith.subf %out, %22 : f64
          %24 = linalg.index 0 : index
          %25 = arith.cmpi slt, %24, %arg8 : index
          %26 = arith.select %25, %23, %out : f64
          linalg.yield %26 : f64
        } -> tensor<?xf64>
        %20 = polygeist.submapInverse(%inserted_0, %19, %15) {map = #map1} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
        %extracted_1 = tensor.extract %20[] : tensor<f64>
        %extracted_2 = tensor.extract %arg10[%arg8, %arg8] : tensor<?x40xf64>
        %21 = arith.divf %extracted_1, %extracted_2 : f64
        %inserted_3 = tensor.insert %21 into %arg10[%arg5, %arg8] : tensor<?x40xf64>
        affine.yield %20, %inserted_3 : tensor<f64>, tensor<?x40xf64>
      }
      %14:2 = affine.for %arg8 = #map(%arg5) to %4 iter_args(%arg9 = %13#0, %arg10 = %13#1) -> (tensor<f64>, tensor<?x40xf64>) {
        %extracted = tensor.extract %arg10[%arg5, %arg8] : tensor<?x40xf64>
        %inserted_0 = tensor.insert %extracted into %arg9[] : tensor<f64>
        %15 = arith.subi %4, %c1 : index
        %16 = polygeist.submap(%inserted_0, %15) {map = #map1} : (tensor<f64>, index) -> tensor<?xf64>
        %17 = polygeist.submap(%arg10, %arg5, %15) {map = #map2} : (tensor<?x40xf64>, index, index) -> tensor<?xf64>
        %18 = polygeist.submap(%arg10, %arg8, %15) {map = #map3} : (tensor<?x40xf64>, index, index) -> tensor<?xf64>
        %19 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%17, %18 : tensor<?xf64>, tensor<?xf64>) outs(%16 : tensor<?xf64>) {
        ^bb0(%in: f64, %in_3: f64, %out: f64):
          %21 = arith.mulf %in, %in_3 : f64
          %22 = arith.subf %out, %21 : f64
          %23 = linalg.index 0 : index
          %24 = arith.cmpi slt, %23, %arg5 : index
          %25 = arith.select %24, %22, %out : f64
          linalg.yield %25 : f64
        } -> tensor<?xf64>
        %20 = polygeist.submapInverse(%inserted_0, %19, %15) {map = #map1} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
        %extracted_1 = tensor.extract %20[] : tensor<f64>
        %inserted_2 = tensor.insert %extracted_1 into %arg10[%arg5, %arg8] : tensor<?x40xf64>
        affine.yield %20, %inserted_2 : tensor<f64>, tensor<?x40xf64>
      }
      affine.yield %14#0, %14#1 : tensor<f64>, tensor<?x40xf64>
    }
    %8 = bufferization.to_memref %7#1 : memref<?x40xf64>
    memref.copy %8, %arg1 : memref<?x40xf64> to memref<?x40xf64>
    %9:2 = affine.for %arg5 = 0 to %4 iter_args(%arg6 = %7#0, %arg7 = %0) -> (tensor<f64>, tensor<?xf64>) {
      %extracted = tensor.extract %2[%arg5] : tensor<?xf64>
      %inserted_0 = tensor.insert %extracted into %arg6[] : tensor<f64>
      %13 = arith.subi %4, %c1 : index
      %14 = polygeist.submap(%inserted_0, %13) {map = #map1} : (tensor<f64>, index) -> tensor<?xf64>
      %15 = polygeist.submap(%7#1, %arg5, %13) {map = #map2} : (tensor<?x40xf64>, index, index) -> tensor<?xf64>
      %16 = polygeist.submap(%arg7, %13) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%15, %16 : tensor<?xf64>, tensor<?xf64>) outs(%14 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_3: f64, %out: f64):
        %19 = arith.mulf %in, %in_3 : f64
        %20 = arith.subf %out, %19 : f64
        %21 = linalg.index 0 : index
        %22 = arith.cmpi slt, %21, %arg5 : index
        %23 = arith.select %22, %20, %out : f64
        linalg.yield %23 : f64
      } -> tensor<?xf64>
      %18 = polygeist.submapInverse(%inserted_0, %17, %13) {map = #map1} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
      %extracted_1 = tensor.extract %18[] : tensor<f64>
      %inserted_2 = tensor.insert %extracted_1 into %arg7[%arg5] : tensor<?xf64>
      affine.yield %18, %inserted_2 : tensor<f64>, tensor<?xf64>
    }
    %10 = bufferization.to_memref %9#1 : memref<?xf64>
    memref.copy %10, %arg4 : memref<?xf64> to memref<?xf64>
    %11:2 = affine.for %arg5 = 0 to %4 iter_args(%arg6 = %9#0, %arg7 = %1) -> (tensor<f64>, tensor<?xf64>) {
      %13 = affine.apply #map4(%arg5)[%4]
      %extracted = tensor.extract %9#1[%13] : tensor<?xf64>
      %inserted_0 = tensor.insert %extracted into %arg6[] : tensor<f64>
      %14 = polygeist.submap(%inserted_0, %4) {map = #map1} : (tensor<f64>, index) -> tensor<?xf64>
      %15 = polygeist.submap(%7#1, %arg5, %4, %4) {map = #map5} : (tensor<?x40xf64>, index, index, index) -> tensor<?xf64>
      %16 = polygeist.submap(%arg7, %4) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["reduction"], library_call = ""} ins(%15, %16 : tensor<?xf64>, tensor<?xf64>) outs(%14 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_4: f64, %out: f64):
        %23 = arith.mulf %in, %in_4 : f64
        %24 = arith.subf %out, %23 : f64
        %25 = linalg.index 0 : index
        %26 = affine.apply #map6(%arg5)[%4]
        %27 = arith.cmpi sge, %25, %26 : index
        %28 = arith.select %27, %24, %out : f64
        linalg.yield %28 : f64
      } -> tensor<?xf64>
      %18 = polygeist.submapInverse(%inserted_0, %17, %4) {map = #map1} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
      %extracted_1 = tensor.extract %18[] : tensor<f64>
      %19 = affine.apply #map4(%arg5)[%4]
      %20 = affine.apply #map4(%arg5)[%4]
      %extracted_2 = tensor.extract %7#1[%19, %20] : tensor<?x40xf64>
      %21 = arith.divf %extracted_1, %extracted_2 : f64
      %22 = affine.apply #map4(%arg5)[%4]
      %inserted_3 = tensor.insert %21 into %arg7[%22] : tensor<?xf64>
      affine.yield %18, %inserted_3 : tensor<f64>, tensor<?xf64>
    }
    %12 = bufferization.to_memref %11#1 : memref<?xf64>
    memref.copy %12, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}

