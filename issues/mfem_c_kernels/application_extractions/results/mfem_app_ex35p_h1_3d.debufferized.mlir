#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d2 * 5 + d0 * 750)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 250)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 375)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 500)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 625)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 125 + d1 * 25 + d2 * 5)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex35p_h1_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg7 : memref<?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?xf64>
    %2 = bufferization.to_tensor %arg5 : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 : memref<?xf64>
    %4 = bufferization.to_tensor %arg3 : memref<?xf64>
    %5 = bufferization.to_tensor %arg2 : memref<?xf64>
    %6 = bufferization.to_tensor %arg1 : memref<?xf64>
    %7 = bufferization.to_tensor %arg0 : memref<?xf64>
    %8 = tensor.empty() : tensor<2x4x4x4xf64>
    %9 = tensor.empty() : tensor<2x4x4x4xf64>
    %10 = tensor.empty() : tensor<2x4x4x4xf64>
    %11 = tensor.empty() : tensor<2x5x4x4xf64>
    %12 = tensor.empty() : tensor<2x5x4x4xf64>
    %13 = tensor.empty() : tensor<2x5x4x4xf64>
    %14 = tensor.empty() : tensor<2x5x5x4xf64>
    %15 = tensor.empty() : tensor<2x5x5x4xf64>
    %16 = tensor.empty() : tensor<2x5x5x4xf64>
    %17 = tensor.empty() : tensor<2x5x5x5xf64>
    %18 = tensor.empty() : tensor<2x5x5x5xf64>
    %19 = tensor.empty() : tensor<2x5x5x5xf64>
    %20 = tensor.empty() : tensor<2x4x5x5xf64>
    %21 = tensor.empty() : tensor<2x4x5x5xf64>
    %22 = tensor.empty() : tensor<2x4x5x5xf64>
    %23 = tensor.empty() : tensor<2x4x4x5xf64>
    %24 = tensor.empty() : tensor<2x4x4x5xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%24 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %26 = polygeist.submap(%7, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %27 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %28 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%27, %26 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%25 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x4x5xf64>
    %29 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%23 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %30 = polygeist.submap(%6, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %31 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %32 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%31, %30 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%29 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x4x5xf64>
    %33 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%22 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %34 = polygeist.submap(%7, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %35 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%32, %34 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%33 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x5x5xf64>
    %36 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%21 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %37 = polygeist.submap(%6, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %38 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%28, %37 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%36 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x5x5xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%20 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %40 = polygeist.submap(%7, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %41 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%28, %40 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%39 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x5x5xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%19 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %43 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %44 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%35, %43 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%42 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x5x5xf64>
    %45 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%18 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %46 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%38, %46 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%45 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x5x5xf64>
    %48 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%17 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %49 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%41, %49 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%48 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x5x5xf64>
    %51 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%16 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %52 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%53, %44, %54, %47, %55, %50, %52 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%51 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.mulf %in_1, %in_2 : f64
      %120 = arith.addf %118, %119 : f64
      %121 = arith.mulf %in_3, %in_4 : f64
      %122 = arith.addf %120, %121 : f64
      %123 = arith.mulf %122, %in_5 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    } -> tensor<2x5x5x4xf64>
    %57 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%15 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %58 = polygeist.submap(%5, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %59 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %60 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%59, %44, %60, %47, %61, %50, %58 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%57 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.mulf %in_1, %in_2 : f64
      %120 = arith.addf %118, %119 : f64
      %121 = arith.mulf %in_3, %in_4 : f64
      %122 = arith.addf %120, %121 : f64
      %123 = arith.mulf %122, %in_5 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    } -> tensor<2x5x5x4xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%14 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %64 = polygeist.submap(%5, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %65 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %66 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %67 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %68 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%65, %44, %66, %47, %67, %50, %64 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%63 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.mulf %in_1, %in_2 : f64
      %120 = arith.addf %118, %119 : f64
      %121 = arith.mulf %in_3, %in_4 : f64
      %122 = arith.addf %120, %121 : f64
      %123 = arith.mulf %122, %in_5 : f64
      %124 = arith.addf %out, %123 : f64
      linalg.yield %124 : f64
    } -> tensor<2x5x5x4xf64>
    %69 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %70 = polygeist.submap(%5, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%56, %70 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%69 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x4x4xf64>
    %72 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%12 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %73 = polygeist.submap(%4, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %74 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%62, %73 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%72 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x4x4xf64>
    %75 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%11 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %76 = polygeist.submap(%5, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %77 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%68, %76 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%75 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x4x4xf64>
    %78 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %79 = polygeist.submap(%5, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %80 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%71, %79 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%78 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x4x4xf64>
    %81 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%9 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %82 = polygeist.submap(%5, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %83 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%74, %82 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%81 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x4x4xf64>
    %84 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %85 = polygeist.submap(%4, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%77, %85 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%84 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x4x4xf64>
    %87 = polygeist.submap(%0, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %88 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%80, %83, %86 : tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>) outs(%87 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %118 = arith.addf %in, %in_0 : f64
      %119 = arith.addf %118, %in_1 : f64
      %120 = arith.addf %out, %119 : f64
      linalg.yield %120 : f64
    } -> tensor<?x?x?x?xf64>
    %89 = polygeist.submapInverse(%0, %88, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %90 = tensor.empty() : tensor<2x5x4x4xf64>
    %91 = tensor.empty() : tensor<2x5x5x4xf64>
    %92 = tensor.empty() : tensor<2x5x5x5xf64>
    %93 = tensor.empty() : tensor<2x4x5x5xf64>
    %94 = tensor.empty() : tensor<2x4x4x5xf64>
    %95 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%94 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %96 = polygeist.submap(%7, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %97 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %98 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%96, %97 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%95 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x4x5xf64>
    %99 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%93 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %100 = polygeist.submap(%7, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %101 = linalg.generic {doc = "", indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%100, %98 : tensor<?x?x?x?x?xf64>, tensor<2x4x4x5xf64>) outs(%99 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x4x5x5xf64>
    %102 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%92 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %103 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %104 = linalg.generic {doc = "", indexing_maps = [#map3, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%103, %101 : tensor<?x?x?x?x?xf64>, tensor<2x4x5x5xf64>) outs(%102 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x5x5xf64>
    %105 = polygeist.submap(%2, %c2, %c5, %c5, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %106 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%105 : tensor<?x?x?x?xf64>) outs(%104 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %out: f64):
      %118 = arith.mulf %out, %in : f64
      linalg.yield %118 : f64
    } -> tensor<2x5x5x5xf64>
    %107 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%91 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %108 = polygeist.submap(%5, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %109 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%108, %106 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>) outs(%107 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x5x4xf64>
    %110 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%90 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %111 = polygeist.submap(%5, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %112 = linalg.generic {doc = "", indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%111, %109 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x4xf64>) outs(%110 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<2x5x4x4xf64>
    %113 = polygeist.submap(%5, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %114 = polygeist.submap(%89, %c2, %c4, %c4, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %115 = linalg.generic {doc = "", indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%113, %112 : tensor<?x?x?x?x?xf64>, tensor<2x5x4x4xf64>) outs(%114 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %118 = arith.mulf %in, %in_0 : f64
      %119 = arith.addf %out, %118 : f64
      linalg.yield %119 : f64
    } -> tensor<?x?x?x?x?xf64>
    %116 = polygeist.submapInverse(%89, %115, %c2, %c4, %c4, %c4, %c5) {map = #map21} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %117 = bufferization.to_memref %116 : memref<?xf64>
    memref.copy %117, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
}
