#map = affine_map<(d0, d1)[s0] -> (-d1 + s0 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map2 = affine_map<(d0)[s0] -> (s0, d0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map5 = affine_map<(d0) -> ()>
#map6 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ll(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = affine.for %arg10 = 0 to %4 iter_args(%arg11 = %0) -> (tensor<?x?xf64>) {
      %7 = affine.for %arg12 = 0 to %3 iter_args(%arg13 = %arg11) -> (tensor<?x?xf64>) {
        %8 = affine.apply #map(%arg10, %arg12)[%3]
        %extracted = tensor.extract %1[%arg10, %8] : tensor<?x?xf64>
        %9 = arith.mulf %arg2, %extracted : f64
        %alloca = memref.alloca() : memref<f64>
        %10 = bufferization.to_tensor %alloca : memref<f64>
        %inserted = tensor.insert %cst into %10[] : tensor<f64>
        %11 = polygeist.submap(%2, %arg12, %3, %3) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
        %12 = polygeist.submap(%arg13, %arg10, %3) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %13 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%11 : tensor<?xf64>) outs(%12 : tensor<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          %27 = arith.mulf %9, %in : f64
          %28 = arith.addf %out, %27 : f64
          %29 = linalg.index 0 : index
          %30 = affine.apply #map4(%arg12)[%3]
          %31 = arith.cmpi sge, %29, %30 : index
          %32 = arith.select %31, %28, %out : f64
          linalg.yield %32 : f64
        } -> tensor<?xf64>
        %14 = polygeist.submapInverse(%arg13, %13, %arg10, %3) {map = #map2} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
        %15 = polygeist.submap(%1, %arg10, %3) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %16 = polygeist.submap(%2, %arg12, %3, %3) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
        %17 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map5], iterator_types = ["reduction"], library_call = ""} ins(%15, %16 : tensor<?xf64>, tensor<?xf64>) outs(%inserted : tensor<f64>) {
        ^bb0(%in: f64, %in_4: f64, %out: f64):
          %27 = arith.mulf %in, %in_4 : f64
          %28 = arith.addf %out, %27 : f64
          %29 = linalg.index 0 : index
          %30 = affine.apply #map4(%arg12)[%3]
          %31 = arith.cmpi sge, %29, %30 : index
          %32 = arith.select %31, %28, %out : f64
          linalg.yield %32 : f64
        } -> tensor<f64>
        %extracted_0 = tensor.extract %17[] : tensor<f64>
        %18 = affine.apply #map(%arg10, %arg12)[%3]
        %extracted_1 = tensor.extract %14[%arg10, %18] : tensor<?x?xf64>
        %19 = arith.mulf %arg7, %extracted_1 : f64
        %20 = affine.apply #map6(%arg12)[%3]
        %21 = affine.apply #map6(%arg12)[%3]
        %extracted_2 = tensor.extract %2[%20, %21] : tensor<?x?xf64>
        %22 = arith.mulf %9, %extracted_2 : f64
        %23 = arith.addf %19, %22 : f64
        %24 = arith.mulf %arg2, %extracted_0 : f64
        %25 = arith.addf %23, %24 : f64
        %26 = affine.apply #map(%arg10, %arg12)[%3]
        %inserted_3 = tensor.insert %25 into %14[%arg10, %26] : tensor<?x?xf64>
        affine.yield %inserted_3 : tensor<?x?xf64>
      }
      affine.yield %7 : tensor<?x?xf64>
    }
    %6 = bufferization.to_memref %5 : memref<?x?xf64>
    memref.copy %6, %arg9 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

