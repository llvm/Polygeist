#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5 + d0 * 75)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 75 + d1 * 5 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 75 + d1 * 5 + 50)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map13 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg6 : memref<?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg2 : memref<?xf64>
    %5 = bufferization.to_tensor %arg1 : memref<?xf64>
    %6 = bufferization.to_tensor %arg0 : memref<?xf64>
    %7 = tensor.empty() : tensor<2x4x4xf64>
    %8 = tensor.empty() : tensor<2x4x4xf64>
    %9 = tensor.empty() : tensor<2x5x4xf64>
    %10 = tensor.empty() : tensor<2x5x4xf64>
    %11 = tensor.empty() : tensor<2x5x5xf64>
    %12 = tensor.empty() : tensor<2x5x5xf64>
    %13 = tensor.empty() : tensor<2x4x5xf64>
    %14 = tensor.empty() : tensor<2x4x5xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%14 : tensor<2x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5xf64>
    %16 = polygeist.submap(%6, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %17 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%17, %16 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%15 : tensor<2x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<2x4x5xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<2x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5xf64>
    %20 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %21 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%21, %20 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%19 : tensor<2x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<2x4x5xf64>
    %23 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%12 : tensor<2x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5xf64>
    %24 = polygeist.submap(%6, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%22, %24 : tensor<2x4x5xf64>, tensor<?x?x?x?xf64>) outs(%23 : tensor<2x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<2x5x5xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%11 : tensor<2x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5xf64>
    %27 = polygeist.submap(%5, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %28 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%18, %27 : tensor<2x4x5xf64>, tensor<?x?x?x?xf64>) outs(%26 : tensor<2x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<2x5x5xf64>
    %29 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<2x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4xf64>
    %30 = polygeist.submap(%3, %c2, %c5, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %31 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %32 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %33 = linalg.generic {doc = "", indexing_maps = [#map3, #map10, #map3, #map10, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%31, %25, %32, %28, %30 : tensor<?x?x?x?xf64>, tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, tensor<2x5x5xf64>, tensor<?x?x?x?xf64>) outs(%29 : tensor<2x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.mulf %in_1, %in_2 : f64
      %51 = arith.addf %49, %50 : f64
      %52 = arith.mulf %51, %in_3 : f64
      %53 = arith.addf %out, %52 : f64
      linalg.yield %53 : f64
    } -> tensor<2x5x4xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%9 : tensor<2x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4xf64>
    %35 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %36 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %37 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %38 = linalg.generic {doc = "", indexing_maps = [#map3, #map10, #map3, #map10, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%36, %25, %37, %28, %35 : tensor<?x?x?x?xf64>, tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, tensor<2x5x5xf64>, tensor<?x?x?x?xf64>) outs(%34 : tensor<2x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.mulf %in_1, %in_2 : f64
      %51 = arith.addf %49, %50 : f64
      %52 = arith.mulf %51, %in_3 : f64
      %53 = arith.addf %out, %52 : f64
      linalg.yield %53 : f64
    } -> tensor<2x5x4xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<2x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4xf64>
    %40 = polygeist.submap(%4, %c2, %c4, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %41 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%33, %40 : tensor<2x5x4xf64>, tensor<?x?x?x?xf64>) outs(%39 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<2x4x4xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%7 : tensor<2x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4xf64>
    %43 = polygeist.submap(%3, %c2, %c4, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %44 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%38, %43 : tensor<2x5x4xf64>, tensor<?x?x?x?xf64>) outs(%42 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.mulf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<2x4x4xf64>
    %45 = polygeist.submap(%0, %c2, %c4, %c4) {map = #map13} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %46 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%41, %44 : tensor<2x4x4xf64>, tensor<2x4x4xf64>) outs(%45 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %49 = arith.addf %in, %in_0 : f64
      %50 = arith.addf %out, %49 : f64
      linalg.yield %50 : f64
    } -> tensor<?x?x?xf64>
    %47 = polygeist.submapInverse(%0, %46, %c2, %c4, %c4) {map = #map13} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %48 = bufferization.to_memref %47 : memref<?xf64>
    memref.copy %48, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}
