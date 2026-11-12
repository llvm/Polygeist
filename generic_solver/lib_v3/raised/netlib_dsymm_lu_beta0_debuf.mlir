#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_lu_beta0(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: i32, %arg8: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg8 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.subi %4, %c1 : index
    %6 = arith.subi %4, %c1 : index
    %7 = affine.for %arg9 = 0 to %3 iter_args(%arg10 = %0) -> (tensor<?x?xf64>) {
      %9 = affine.for %arg11 = 0 to %4 iter_args(%arg12 = %arg10) -> (tensor<?x?xf64>) {
        %extracted = tensor.extract %1[%arg9, %arg11] : tensor<?x?xf64>
        %10 = arith.mulf %arg2, %extracted : f64
        %alloca = memref.alloca() : memref<f64>
        %11 = bufferization.to_tensor %alloca : memref<f64>
        %inserted = tensor.insert %cst into %11[] : tensor<f64>
        %12 = polygeist.submap(%2, %arg11, %5) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %13 = polygeist.submap(%arg12, %arg9, %5) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %14 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%12 : tensor<?xf64>) outs(%13 : tensor<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          %22 = arith.mulf %10, %in : f64
          %23 = arith.addf %out, %22 : f64
          %24 = linalg.index 0 : index
          %25 = arith.cmpi slt, %24, %arg11 : index
          %26 = arith.select %25, %23, %out : f64
          linalg.yield %26 : f64
        } -> tensor<?xf64>
        %15 = polygeist.submapInverse(%arg12, %14, %arg9, %5) {map = #map} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
        %16 = polygeist.submap(%1, %arg9, %6) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %17 = polygeist.submap(%2, %arg11, %6) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %18 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"], library_call = ""} ins(%16, %17 : tensor<?xf64>, tensor<?xf64>) outs(%inserted : tensor<f64>) {
        ^bb0(%in: f64, %in_3: f64, %out: f64):
          %22 = arith.mulf %in, %in_3 : f64
          %23 = arith.addf %out, %22 : f64
          %24 = linalg.index 0 : index
          %25 = arith.cmpi slt, %24, %arg11 : index
          %26 = arith.select %25, %23, %out : f64
          linalg.yield %26 : f64
        } -> tensor<f64>
        %extracted_0 = tensor.extract %18[] : tensor<f64>
        %extracted_1 = tensor.extract %2[%arg11, %arg11] : tensor<?x?xf64>
        %19 = arith.mulf %10, %extracted_1 : f64
        %20 = arith.mulf %arg2, %extracted_0 : f64
        %21 = arith.addf %19, %20 : f64
        %inserted_2 = tensor.insert %21 into %15[%arg9, %arg11] : tensor<?x?xf64>
        affine.yield %inserted_2 : tensor<?x?xf64>
      }
      affine.yield %9 : tensor<?x?xf64>
    }
    %8 = bufferization.to_memref %7 : memref<?x?xf64>
    memref.copy %8, %arg8 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

