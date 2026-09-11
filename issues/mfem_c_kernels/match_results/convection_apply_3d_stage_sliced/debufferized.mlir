#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 375 + d1 * 25 + d2 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 375 + d1 * 25 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 375 + d1 * 25 + d2 * 5 + 250)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %6 = tensor.empty() : tensor<2x4x4x4xf64>
    %7 = tensor.empty() : tensor<2x5x4x4xf64>
    %8 = tensor.empty() : tensor<2x5x5x4xf64>
    %9 = tensor.empty() : tensor<2x5x5x5xf64>
    %10 = tensor.empty() : tensor<2x5x5x5xf64>
    %11 = tensor.empty() : tensor<2x5x5x5xf64>
    %12 = tensor.empty() : tensor<2x4x5x5xf64>
    %13 = tensor.empty() : tensor<2x4x5x5xf64>
    %14 = tensor.empty() : tensor<2x4x5x5xf64>
    %15 = tensor.empty() : tensor<2x4x4x5xf64>
    %16 = tensor.empty() : tensor<2x4x4x5xf64>
    %17 = polygeist.submap(%16, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%17 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %19 = polygeist.submapInverse(%16, %18, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %20 = polygeist.submap(%4, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %21 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %22 = polygeist.submap(%19, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %23 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%20, %21 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%22 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %24 = polygeist.submapInverse(%19, %23, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %25 = polygeist.submap(%15, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%25 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %27 = polygeist.submapInverse(%15, %26, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %28 = polygeist.submap(%4, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %29 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%27, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %31 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%28, %29 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%30 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %32 = polygeist.submapInverse(%27, %31, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %33 = polygeist.submap(%14, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%14, %34, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %36 = polygeist.submap(%32, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %38 = polygeist.submap(%35, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%36, %37 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%38 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %40 = polygeist.submapInverse(%35, %39, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %41 = polygeist.submap(%13, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%41 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %43 = polygeist.submapInverse(%13, %42, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %44 = polygeist.submap(%24, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%43, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%44, %45 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%46 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %48 = polygeist.submapInverse(%43, %47, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %49 = polygeist.submap(%12, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%49 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %51 = polygeist.submapInverse(%12, %50, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %52 = polygeist.submap(%24, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%51, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%52, %53 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%54 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %56 = polygeist.submapInverse(%51, %55, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %57 = polygeist.submap(%11, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %58 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%57 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %59 = polygeist.submapInverse(%11, %58, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %60 = polygeist.submap(%40, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%59, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%60, %61 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%62 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %64 = polygeist.submapInverse(%59, %63, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %65 = polygeist.submap(%10, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %66 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%65 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %67 = polygeist.submapInverse(%10, %66, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %68 = polygeist.submap(%48, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %69 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = polygeist.submap(%67, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%68, %69 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%70 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %72 = polygeist.submapInverse(%67, %71, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %73 = polygeist.submap(%9, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %74 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%73 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %75 = polygeist.submapInverse(%9, %74, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %76 = polygeist.submap(%56, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %77 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %78 = polygeist.submap(%75, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %79 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%76, %77 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%78 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %80 = polygeist.submapInverse(%75, %79, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %81 = polygeist.submap(%8, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %82 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%81 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %83 = polygeist.submapInverse(%8, %82, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %84 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %85 = polygeist.submap(%64, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %87 = polygeist.submap(%72, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %88 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %89 = polygeist.submap(%80, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %90 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %91 = polygeist.submap(%83, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %92 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%84, %85, %86, %87, %88, %89, %90 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%91 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.mulf %in_1, %in_2 : f64
      %117 = arith.addf %115, %116 : f64
      %118 = arith.mulf %in_3, %in_4 : f64
      %119 = arith.addf %117, %118 : f64
      %120 = arith.mulf %119, %in_5 : f64
      %121 = arith.addf %out, %120 : f64
      linalg.yield %121 : f64
    } -> tensor<?x?x?x?x?xf64>
    %93 = polygeist.submapInverse(%83, %92, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %94 = polygeist.submap(%7, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %95 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%94 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %96 = polygeist.submapInverse(%7, %95, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %97 = polygeist.submap(%93, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %98 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %99 = polygeist.submap(%96, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %100 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%97, %98 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%99 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %101 = polygeist.submapInverse(%96, %100, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %102 = polygeist.submap(%6, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %103 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%102 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %104 = polygeist.submapInverse(%6, %103, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %105 = polygeist.submap(%101, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %106 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %107 = polygeist.submap(%104, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %108 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%105, %106 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%107 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %115 = arith.mulf %in, %in_0 : f64
      %116 = arith.addf %out, %115 : f64
      linalg.yield %116 : f64
    } -> tensor<?x?x?x?x?xf64>
    %109 = polygeist.submapInverse(%104, %108, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %110 = polygeist.submap(%109, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %111 = polygeist.submap(%5, %c2, %c4, %c4, %c4) {map = #map16} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %112 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%110 : tensor<?x?x?x?xf64>) outs(%111 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %115 = arith.addf %out, %in : f64
      linalg.yield %115 : f64
    } -> tensor<?x?x?x?xf64>
    %113 = polygeist.submapInverse(%5, %112, %c2, %c4, %c4, %c4) {map = #map16} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %114 = bufferization.to_memref %113 : memref<?xf64>
    memref.copy %114, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}
