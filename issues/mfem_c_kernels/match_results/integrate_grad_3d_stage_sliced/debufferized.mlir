#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d1 + d0 * 375 + d2 * 5)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d1 + d0 * 375 + d2 * 5 + 125)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d1 + d0 * 375 + d2 * 5 + 250)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d2)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_grad_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = tensor.empty() : tensor<2x4x4x4xf64>
    %5 = tensor.empty() : tensor<2x4x4x4xf64>
    %6 = tensor.empty() : tensor<2x4x4x4xf64>
    %7 = tensor.empty() : tensor<2x5x4x4xf64>
    %8 = tensor.empty() : tensor<2x5x4x4xf64>
    %9 = tensor.empty() : tensor<2x5x4x4xf64>
    %10 = tensor.empty() : tensor<2x5x5x4xf64>
    %11 = tensor.empty() : tensor<2x5x5x4xf64>
    %12 = tensor.empty() : tensor<2x5x5x4xf64>
    %13 = polygeist.submap(%12, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %14 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %15 = polygeist.submapInverse(%12, %14, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %16 = polygeist.submap(%0, %c2, %c5, %c5, %c4, %c5) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %17 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %18 = polygeist.submap(%15, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%16, %17 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%18 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %20 = polygeist.submapInverse(%15, %19, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %21 = polygeist.submap(%11, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%21 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %23 = polygeist.submapInverse(%11, %22, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %24 = polygeist.submap(%0, %c2, %c5, %c5, %c4, %c5) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %25 = polygeist.submap(%1, %c2, %c5, %c5, %c4, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %26 = polygeist.submap(%23, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%24, %25 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%26 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %28 = polygeist.submapInverse(%23, %27, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %29 = polygeist.submap(%10, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %30 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%29 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %31 = polygeist.submapInverse(%10, %30, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %32 = polygeist.submap(%0, %c2, %c5, %c5, %c4, %c5) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %33 = polygeist.submap(%1, %c2, %c5, %c5, %c4, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %34 = polygeist.submap(%31, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %35 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%32, %33 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%34 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %36 = polygeist.submapInverse(%31, %35, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %37 = polygeist.submap(%9, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %38 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%37 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %39 = polygeist.submapInverse(%9, %38, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %40 = polygeist.submap(%20, %c2, %c5, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %41 = polygeist.submap(%1, %c2, %c5, %c4, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %42 = polygeist.submap(%39, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %43 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%40, %41 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%42 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %44 = polygeist.submapInverse(%39, %43, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %45 = polygeist.submap(%8, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %46 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%45 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %47 = polygeist.submapInverse(%8, %46, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %48 = polygeist.submap(%28, %c2, %c5, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %49 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %50 = polygeist.submap(%47, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %51 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%48, %49 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%50 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %52 = polygeist.submapInverse(%47, %51, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %53 = polygeist.submap(%7, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %54 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%53 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %55 = polygeist.submapInverse(%7, %54, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %56 = polygeist.submap(%36, %c2, %c5, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %57 = polygeist.submap(%1, %c2, %c5, %c4, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %58 = polygeist.submap(%55, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %59 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%56, %57 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%58 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %60 = polygeist.submapInverse(%55, %59, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %61 = polygeist.submap(%6, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %62 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%61 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %63 = polygeist.submapInverse(%6, %62, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %64 = polygeist.submap(%44, %c2, %c4, %c4, %c4, %c5) {map = #map9} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %65 = polygeist.submap(%1, %c2, %c4, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %66 = polygeist.submap(%63, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %67 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%64, %65 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%66 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %68 = polygeist.submapInverse(%63, %67, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %69 = polygeist.submap(%5, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %70 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%69 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %71 = polygeist.submapInverse(%5, %70, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %72 = polygeist.submap(%52, %c2, %c4, %c4, %c4, %c5) {map = #map9} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %73 = polygeist.submap(%1, %c2, %c4, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %74 = polygeist.submap(%71, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %75 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%72, %73 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%74 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %76 = polygeist.submapInverse(%71, %75, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %77 = polygeist.submap(%4, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %78 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%77 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %79 = polygeist.submapInverse(%4, %78, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %80 = polygeist.submap(%60, %c2, %c4, %c4, %c4, %c5) {map = #map9} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %81 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %82 = polygeist.submap(%79, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %83 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%80, %81 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%82 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %92 = arith.mulf %in, %in_0 : f64
      %93 = arith.addf %out, %92 : f64
      linalg.yield %93 : f64
    } -> tensor<?x?x?x?x?xf64>
    %84 = polygeist.submapInverse(%79, %83, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %85 = polygeist.submap(%68, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %86 = polygeist.submap(%76, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %87 = polygeist.submap(%84, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %88 = polygeist.submap(%3, %c2, %c4, %c4, %c4) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %89 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%85, %86, %87 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%88 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %92 = arith.addf %in, %in_0 : f64
      %93 = arith.addf %92, %in_1 : f64
      %94 = arith.addf %out, %93 : f64
      linalg.yield %94 : f64
    } -> tensor<?x?x?x?xf64>
    %90 = polygeist.submapInverse(%3, %89, %c2, %c4, %c4, %c4) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %91 = bufferization.to_memref %90 : memref<?xf64>
    memref.copy %91, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}
