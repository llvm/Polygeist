#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_ssyr2k_ut(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: i32, %arg4: memref<?x?xf32>, %arg5: i32, %arg6: memref<?x?xf32>, %arg7: f32, %arg8: i32, %arg9: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg9 : memref<?x?xf32>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf32>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = affine.for %arg10 = 0 to %3 iter_args(%arg11 = %0) -> (tensor<?x?xf32>) {
      %7 = affine.for %arg12 = 0 to #map(%arg10) iter_args(%arg13 = %arg11) -> (tensor<?x?xf32>) {
        %alloca = memref.alloca() : memref<f32>
        %8 = bufferization.to_tensor %alloca : memref<f32>
        %inserted = tensor.insert %cst into %8[] : tensor<f32>
        %alloca_0 = memref.alloca() : memref<f32>
        %9 = bufferization.to_tensor %alloca_0 : memref<f32>
        %inserted_1 = tensor.insert %cst into %9[] : tensor<f32>
        %10 = polygeist.submap(%1, %arg12, %4) {map = #map1} : (tensor<?x?xf32>, index, index) -> tensor<?xf32>
        %11 = polygeist.submap(%2, %arg10, %4) {map = #map1} : (tensor<?x?xf32>, index, index) -> tensor<?xf32>
        %12 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map3], iterator_types = ["reduction"], library_call = ""} ins(%10, %11 : tensor<?xf32>, tensor<?xf32>) outs(%inserted : tensor<f32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %21 = arith.mulf %in, %in_5 : f32
          %22 = arith.addf %out, %21 : f32
          linalg.yield %22 : f32
        } -> tensor<f32>
        %13 = polygeist.submap(%2, %arg12, %4) {map = #map1} : (tensor<?x?xf32>, index, index) -> tensor<?xf32>
        %14 = polygeist.submap(%1, %arg10, %4) {map = #map1} : (tensor<?x?xf32>, index, index) -> tensor<?xf32>
        %15 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map3], iterator_types = ["reduction"], library_call = ""} ins(%13, %14 : tensor<?xf32>, tensor<?xf32>) outs(%inserted_1 : tensor<f32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %21 = arith.mulf %in, %in_5 : f32
          %22 = arith.addf %out, %21 : f32
          linalg.yield %22 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %12[] : tensor<f32>
        %extracted_2 = tensor.extract %15[] : tensor<f32>
        %extracted_3 = tensor.extract %arg13[%arg10, %arg12] : tensor<?x?xf32>
        %16 = arith.mulf %arg7, %extracted_3 : f32
        %17 = arith.mulf %arg2, %extracted_2 : f32
        %18 = arith.addf %16, %17 : f32
        %19 = arith.mulf %arg2, %extracted : f32
        %20 = arith.addf %18, %19 : f32
        %inserted_4 = tensor.insert %20 into %arg13[%arg10, %arg12] : tensor<?x?xf32>
        affine.yield %inserted_4 : tensor<?x?xf32>
      }
      affine.yield %7 : tensor<?x?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?x?xf32>
    memref.copy %6, %arg9 : memref<?x?xf32> to memref<?x?xf32>
    return
  }
}

