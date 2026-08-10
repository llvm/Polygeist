#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2 + d0 * 25 + d1 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map11 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5 + 25)>
#map14 = affine_map<(d0) -> (d0)>
#map15 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex9p_mass_convection_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: f64, %arg8: f64, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>, %arg12: memref<?xf64>, %arg13: memref<?xf64>, %arg14: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg14 : memref<?xf64>
    %1 = bufferization.to_tensor %arg13 : memref<?xf64>
    %2 = bufferization.to_tensor %arg12 : memref<?xf64>
    %3 = bufferization.to_tensor %arg11 : memref<?xf64>
    %4 = bufferization.to_tensor %arg10 : memref<?xf64>
    %5 = bufferization.to_tensor %arg9 : memref<?xf64>
    %6 = bufferization.to_tensor %arg6 : memref<?xf64>
    %7 = bufferization.to_tensor %arg5 : memref<?xf64>
    %8 = bufferization.to_tensor %arg4 : memref<?xf64>
    %9 = bufferization.to_tensor %arg3 : memref<?xf64>
    %10 = bufferization.to_tensor %arg2 : memref<?xf64>
    %11 = bufferization.to_tensor %arg1 : memref<?xf64>
    %12 = bufferization.to_tensor %arg0 : memref<?xf64>
    %13 = tensor.empty() : tensor<2x5x4xf64>
    %14 = tensor.empty() : tensor<2x5x5xf64>
    %15 = tensor.empty() : tensor<2x4x5xf64>
    %extracted_slice = tensor.extract_slice %15[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %16 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice = tensor.insert_slice %16 into %15[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %17 = polygeist.submap(%12, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %18 = polygeist.submap(%7, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%17, %18 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%inserted_slice : tensor<2x4x5xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<2x4x5xf64>
    %extracted_slice_0 = tensor.extract_slice %14[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : tensor<2x5x5xf64> to tensor<?x?x?xf64>
    %20 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_0 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %19[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %21 = polygeist.submap(%12, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = linalg.generic {doc = "", indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%21, %extracted_slice_1 : tensor<?x?x?x?xf64>, tensor<?x?x?xf64>) outs(%20 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?x?x?xf64>
    %23 = polygeist.submap(%9, %c2, %c5, %c5) {map = #map7} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %24 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%23 : tensor<?x?x?xf64>) outs(%22 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %71 = arith.mulf %out, %in : f64
      linalg.yield %71 : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %13[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_2 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %26 = polygeist.submap(%10, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map3, #map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%26, %24 : tensor<?x?x?x?xf64>, tensor<?x?x?xf64>) outs(%25 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?x?x?xf64>
    %28 = polygeist.submap(%10, %c2, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %29 = polygeist.submap(%5, %c2, %c4, %c4) {map = #map11} : (tensor<?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %30 = linalg.generic {doc = "", indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%28, %27 : tensor<?x?x?x?xf64>, tensor<?x?x?xf64>) outs(%29 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<2x4x4xf64>
    %31 = polygeist.submapInverse(%5, %30, %c2, %c4, %c4) {map = #map11} : (tensor<?xf64>, tensor<2x4x4xf64>, index, index, index) -> tensor<?xf64>
    %32 = bufferization.to_memref %31 : memref<?xf64>
    memref.copy %32, %arg9 : memref<?xf64> to memref<?xf64>
    %33 = tensor.empty() : tensor<2x4x4xf64>
    %34 = tensor.empty() : tensor<2x5x4xf64>
    %35 = tensor.empty() : tensor<2x5x5xf64>
    %36 = tensor.empty() : tensor<2x5x5xf64>
    %37 = tensor.empty() : tensor<2x4x5xf64>
    %38 = tensor.empty() : tensor<2x4x5xf64>
    %extracted_slice_3 = tensor.extract_slice %38[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_3 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_4 = tensor.insert_slice %39 into %38[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %40 = polygeist.submap(%12, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %41 = polygeist.submap(%7, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%41, %40 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%inserted_slice_4 : tensor<2x4x5xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<2x4x5xf64>
    %extracted_slice_5 = tensor.extract_slice %37[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %43 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_5 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_6 = tensor.insert_slice %43 into %37[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %44 = polygeist.submap(%11, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %45 = polygeist.submap(%7, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %46 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%45, %44 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%inserted_slice_6 : tensor<2x4x5xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<2x4x5xf64>
    %extracted_slice_7 = tensor.extract_slice %36[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : tensor<2x5x5xf64> to tensor<?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_7 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_8 = tensor.extract_slice %46[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %48 = polygeist.submap(%12, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %49 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%extracted_slice_8, %48 : tensor<?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%47 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_9 = tensor.extract_slice %35[0, 0, 0] [%c2, %c5, %c5] [1, 1, 1] : tensor<2x5x5xf64> to tensor<?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_9 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_10 = tensor.extract_slice %42[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %51 = polygeist.submap(%11, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %52 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%extracted_slice_10, %51 : tensor<?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%50 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_11 = tensor.extract_slice %34[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %53 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_11 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %54 = polygeist.submap(%10, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %55 = polygeist.submap(%8, %c2, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %56 = polygeist.submap(%8, %c2, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %57 = linalg.generic {doc = "", indexing_maps = [#map3, #map9, #map3, #map9, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%55, %49, %56, %52, %54 : tensor<?x?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%53 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_25: f64, %in_26: f64, %in_27: f64, %in_28: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.mulf %in_26, %in_27 : f64
      %73 = arith.addf %71, %72 : f64
      %74 = arith.mulf %73, %in_28 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?xf64>
    %extracted_slice_12 = tensor.extract_slice %33[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : tensor<2x4x4xf64> to tensor<?x?x?xf64>
    %58 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_12 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %59 = polygeist.submap(%10, %c2, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %60 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%57, %59 : tensor<?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%58 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_25: f64, %out: f64):
      %71 = arith.mulf %in, %in_25 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?x?x?xf64>
    %61 = polygeist.submap(%4, %c2, %c4, %c4) {map = #map11} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %62 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%60 : tensor<?x?x?xf64>) outs(%61 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %71 = arith.addf %out, %in : f64
      linalg.yield %71 : f64
    } -> tensor<?x?x?xf64>
    %63 = polygeist.submapInverse(%4, %62, %c2, %c4, %c4) {map = #map11} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<?xf64>
    %extracted_slice_13 = tensor.extract_slice %6[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_14 = tensor.extract_slice %31[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_15 = tensor.extract_slice %63[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_16 = tensor.extract_slice %3[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_17 = tensor.extract_slice %3[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_18 = tensor.extract_slice %2[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_19 = tensor.extract_slice %1[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_20 = tensor.extract_slice %inserted[0] [1] [1] : tensor<?xf64> to tensor<f64>
    %64:4 = linalg.generic {doc = "", indexing_maps = [#map14, #map14, #map14, #map14, #map14, #map14, #map14, #map15], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_19, %extracted_slice_14, %extracted_slice_13, %extracted_slice_16 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_15, %extracted_slice_17, %extracted_slice_18, %extracted_slice_20 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<f64>) {
    ^bb0(%in: f64, %in_25: f64, %in_26: f64, %in_27: f64, %out: f64, %out_28: f64, %out_29: f64, %out_30: f64):
      %71 = arith.mulf %arg7, %in : f64
      %72 = arith.addf %out, %71 : f64
      %73 = arith.mulf %arg7, %in_25 : f64
      %74 = arith.subf %out_28, %73 : f64
      %75 = arith.mulf %in_26, %74 : f64
      %76 = arith.mulf %in_27, %75 : f64
      %77 = arith.addf %out_30, %76 : f64
      linalg.yield %72, %74, %75, %77 : f64, f64, f64, f64
    } -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<f64>)
    %inserted_slice_21 = tensor.insert_slice %64#3 into %inserted[0] [1] [1] : tensor<f64> into tensor<?xf64>
    %65 = bufferization.to_memref %inserted_slice_21 : memref<?xf64>
    memref.copy %65, %arg14 : memref<?xf64> to memref<?xf64>
    %inserted_slice_22 = tensor.insert_slice %64#2 into %2[0] [%c32] [1] : tensor<?xf64> into tensor<?xf64>
    %66 = bufferization.to_memref %inserted_slice_22 : memref<?xf64>
    memref.copy %66, %arg12 : memref<?xf64> to memref<?xf64>
    %inserted_slice_23 = tensor.insert_slice %64#1 into %3[0] [%c32] [1] : tensor<?xf64> into tensor<?xf64>
    %67 = bufferization.to_memref %inserted_slice_23 : memref<?xf64>
    memref.copy %67, %arg11 : memref<?xf64> to memref<?xf64>
    %inserted_slice_24 = tensor.insert_slice %64#0 into %63[0] [%c32] [1] : tensor<?xf64> into tensor<?xf64>
    %68 = bufferization.to_memref %inserted_slice_24 : memref<?xf64>
    memref.copy %68, %arg10 : memref<?xf64> to memref<?xf64>
    %69 = linalg.generic {doc = "", indexing_maps = [#map14, #map14], iterator_types = ["parallel"], library_call = ""} ins(%inserted_slice_22 : tensor<?xf64>) outs(%1 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %71 = arith.mulf %arg8, %out : f64
      %72 = arith.addf %in, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?xf64>
    %70 = bufferization.to_memref %69 : memref<?xf64>
    memref.copy %70, %arg13 : memref<?xf64> to memref<?xf64>
    return
  }
}

