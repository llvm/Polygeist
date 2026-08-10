#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 3 + d0 * 144)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 4 + d0 * 144 + 48)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 3)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 16 + d2 * 4 + d0 * 144 + 96)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 3)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 125)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 375)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 500)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 250)>
#map22 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5 + 625)>
#map23 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 750 + d1 * 25 + d2 * 5)>
#map24 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 3 + d0 * 144)>
#map25 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 4 + d0 * 144 + 48)>
#map26 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 16 + d2 * 4 + d0 * 144 + 96)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_curlcurl_3d_partial(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg8 : memref<?xf64>
    %1 = bufferization.to_tensor %arg7 : memref<?xf64>
    %2 = bufferization.to_tensor %arg6 : memref<?xf64>
    %3 = bufferization.to_tensor %arg5 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg3 : memref<?xf64>
    %6 = bufferization.to_tensor %arg2 : memref<?xf64>
    %7 = bufferization.to_tensor %arg1 : memref<?xf64>
    %8 = bufferization.to_tensor %arg0 : memref<?xf64>
    %9 = tensor.empty() : tensor<2x4x4x4xf64>
    %10 = tensor.empty() : tensor<2x4x4x4xf64>
    %11 = tensor.empty() : tensor<2x4x4x4xf64>
    %12 = tensor.empty() : tensor<2x4x4x4xf64>
    %13 = tensor.empty() : tensor<2x4x4x4xf64>
    %14 = tensor.empty() : tensor<2x4x4x4xf64>
    %15 = tensor.empty() : tensor<2x5x4x4xf64>
    %16 = tensor.empty() : tensor<2x5x4x4xf64>
    %17 = tensor.empty() : tensor<2x5x4x4xf64>
    %18 = tensor.empty() : tensor<2x5x4x4xf64>
    %19 = tensor.empty() : tensor<2x5x4x4xf64>
    %20 = tensor.empty() : tensor<2x5x4x4xf64>
    %21 = tensor.empty() : tensor<2x5x5x4xf64>
    %22 = tensor.empty() : tensor<2x5x5x4xf64>
    %23 = tensor.empty() : tensor<2x5x5x4xf64>
    %24 = tensor.empty() : tensor<2x5x5x4xf64>
    %25 = tensor.empty() : tensor<2x5x5x4xf64>
    %26 = tensor.empty() : tensor<2x5x5x4xf64>
    %27 = tensor.empty() : tensor<2x5x5x5xf64>
    %28 = tensor.empty() : tensor<2x5x5x5xf64>
    %29 = tensor.empty() : tensor<2x5x5x5xf64>
    %30 = tensor.empty() : tensor<2x5x5x5xf64>
    %31 = tensor.empty() : tensor<2x5x5x5xf64>
    %32 = tensor.empty() : tensor<2x5x5x5xf64>
    %33 = tensor.empty() : tensor<2x4x5x5xf64>
    %34 = tensor.empty() : tensor<2x4x5x5xf64>
    %35 = tensor.empty() : tensor<2x4x5x5xf64>
    %36 = tensor.empty() : tensor<2x4x5x5xf64>
    %37 = tensor.empty() : tensor<2x4x5x5xf64>
    %38 = tensor.empty() : tensor<2x4x5x5xf64>
    %39 = tensor.empty() : tensor<2x4x4x5xf64>
    %40 = tensor.empty() : tensor<2x4x4x5xf64>
    %41 = tensor.empty() : tensor<2x4x4x5xf64>
    %42 = tensor.empty() : tensor<2x4x4x5xf64>
    %43 = tensor.empty() : tensor<2x4x4x5xf64>
    %44 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%43 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %45 = polygeist.submap(%8, %c2, %c4, %c4, %c5, %c3) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c3) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%46, %45 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%44 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x5xf64>
    %48 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%38 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %49 = polygeist.submap(%4, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%47, %49 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%48 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x5x5xf64>
    %51 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%37 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %52 = polygeist.submap(%7, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%47, %52 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%51 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x5x5xf64>
    %54 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%32 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %55 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%50, %55 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%54 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x5x5xf64>
    %57 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%31 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %58 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %59 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%53, %58 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%57 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x5x5xf64>
    %60 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%42 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %61 = polygeist.submap(%4, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%1, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%62, %61 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%60 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x5xf64>
    %64 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%41 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %65 = polygeist.submap(%7, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %66 = polygeist.submap(%1, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %67 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%66, %65 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%64 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x5xf64>
    %68 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%36 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %69 = polygeist.submap(%8, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%63, %69 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%68 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x5x5xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%35 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %72 = polygeist.submap(%8, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %73 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%67, %72 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%71 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x5x5xf64>
    %74 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%30 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %75 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %76 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%70, %75 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%74 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x5x5xf64>
    %77 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%29 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %78 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %79 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%73, %78 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%77 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x5x5xf64>
    %80 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%40 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %81 = polygeist.submap(%4, %c2, %c3, %c4, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %82 = polygeist.submap(%1, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %83 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%82, %81 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%80 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x5xf64>
    %84 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%39 : tensor<2x4x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x5xf64>
    %85 = polygeist.submap(%7, %c2, %c3, %c4, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = polygeist.submap(%1, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %87 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%86, %85 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%84 : tensor<2x4x4x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x5xf64>
    %88 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%34 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %89 = polygeist.submap(%7, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %90 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%83, %89 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%88 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x5x5xf64>
    %91 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%33 : tensor<2x4x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x5x5xf64>
    %92 = polygeist.submap(%4, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %93 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%87, %92 : tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>) outs(%91 : tensor<2x4x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x5x5xf64>
    %94 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%28 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %95 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %96 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%90, %95 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%94 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x5x5xf64>
    %97 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%27 : tensor<2x5x5x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x5xf64>
    %98 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %99 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%93, %98 : tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%97 : tensor<2x5x5x5xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x5x5xf64>
    %100 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%26 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %101 = polygeist.submap(%6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %102 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %103 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %104 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %105 = linalg.generic {doc = "", indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%102, %99, %79, %103, %59, %96, %104, %76, %56, %101 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%100 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %182 = arith.subf %in_0, %in_1 : f64
      %183 = arith.mulf %in, %182 : f64
      %184 = arith.subf %in_3, %in_4 : f64
      %185 = arith.mulf %in_2, %184 : f64
      %186 = arith.addf %183, %185 : f64
      %187 = arith.subf %in_6, %in_7 : f64
      %188 = arith.mulf %in_5, %187 : f64
      %189 = arith.addf %186, %188 : f64
      %190 = arith.mulf %189, %in_8 : f64
      %191 = arith.addf %out, %190 : f64
      linalg.yield %191 : f64
    } -> tensor<2x5x5x4xf64>
    %106 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%20 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %107 = polygeist.submap(%5, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %108 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%105, %107 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%106 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x4x4xf64>
    %109 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%14 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %110 = polygeist.submap(%3, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %111 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%108, %110 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%109 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x4xf64>
    %112 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%25 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %113 = polygeist.submap(%6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %114 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %115 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %116 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %117 = linalg.generic {doc = "", indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%114, %99, %79, %115, %59, %96, %116, %76, %56, %113 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%112 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %182 = arith.subf %in_0, %in_1 : f64
      %183 = arith.mulf %in, %182 : f64
      %184 = arith.subf %in_3, %in_4 : f64
      %185 = arith.mulf %in_2, %184 : f64
      %186 = arith.addf %183, %185 : f64
      %187 = arith.subf %in_6, %in_7 : f64
      %188 = arith.mulf %in_5, %187 : f64
      %189 = arith.addf %186, %188 : f64
      %190 = arith.mulf %189, %in_8 : f64
      %191 = arith.addf %out, %190 : f64
      linalg.yield %191 : f64
    } -> tensor<2x5x5x4xf64>
    %118 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%19 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %119 = polygeist.submap(%3, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %120 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%117, %119 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%118 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x4x4xf64>
    %121 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %122 = polygeist.submap(%5, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %123 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%120, %122 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%121 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x4xf64>
    %124 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%24 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %125 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %126 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %127 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %128 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %129 = linalg.generic {doc = "", indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%126, %99, %79, %127, %59, %96, %128, %76, %56, %125 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%124 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %182 = arith.subf %in_0, %in_1 : f64
      %183 = arith.mulf %in, %182 : f64
      %184 = arith.subf %in_3, %in_4 : f64
      %185 = arith.mulf %in_2, %184 : f64
      %186 = arith.addf %183, %185 : f64
      %187 = arith.subf %in_6, %in_7 : f64
      %188 = arith.mulf %in_5, %187 : f64
      %189 = arith.addf %186, %188 : f64
      %190 = arith.mulf %189, %in_8 : f64
      %191 = arith.addf %out, %190 : f64
      linalg.yield %191 : f64
    } -> tensor<2x5x5x4xf64>
    %130 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%18 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %131 = polygeist.submap(%6, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %132 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%129, %131 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%130 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x4x4xf64>
    %133 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%12 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %134 = polygeist.submap(%5, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %135 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%132, %134 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%133 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x4xf64>
    %136 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%23 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %137 = polygeist.submap(%5, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %138 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map23} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %139 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %140 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %141 = linalg.generic {doc = "", indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%138, %99, %79, %139, %59, %96, %140, %76, %56, %137 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%136 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %182 = arith.subf %in_0, %in_1 : f64
      %183 = arith.mulf %in, %182 : f64
      %184 = arith.subf %in_3, %in_4 : f64
      %185 = arith.mulf %in_2, %184 : f64
      %186 = arith.addf %183, %185 : f64
      %187 = arith.subf %in_6, %in_7 : f64
      %188 = arith.mulf %in_5, %187 : f64
      %189 = arith.addf %186, %188 : f64
      %190 = arith.mulf %189, %in_8 : f64
      %191 = arith.addf %out, %190 : f64
      linalg.yield %191 : f64
    } -> tensor<2x5x5x4xf64>
    %142 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%17 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %143 = polygeist.submap(%6, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %144 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%141, %143 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%142 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x4x4xf64>
    %145 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%11 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %146 = polygeist.submap(%3, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %147 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%144, %146 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%145 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x4xf64>
    %148 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%22 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %149 = polygeist.submap(%5, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %150 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map23} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %151 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %152 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %153 = linalg.generic {doc = "", indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%150, %99, %79, %151, %59, %96, %152, %76, %56, %149 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%148 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %182 = arith.subf %in_0, %in_1 : f64
      %183 = arith.mulf %in, %182 : f64
      %184 = arith.subf %in_3, %in_4 : f64
      %185 = arith.mulf %in_2, %184 : f64
      %186 = arith.addf %183, %185 : f64
      %187 = arith.subf %in_6, %in_7 : f64
      %188 = arith.mulf %in_5, %187 : f64
      %189 = arith.addf %186, %188 : f64
      %190 = arith.mulf %189, %in_8 : f64
      %191 = arith.addf %out, %190 : f64
      linalg.yield %191 : f64
    } -> tensor<2x5x5x4xf64>
    %154 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%16 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %155 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %156 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%153, %155 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%154 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x4x4xf64>
    %157 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %158 = polygeist.submap(%6, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %159 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%156, %158 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%157 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x4xf64>
    %160 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%21 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %161 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %162 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %163 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %164 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %165 = linalg.generic {doc = "", indexing_maps = [#map3, #map18, #map18, #map3, #map18, #map18, #map3, #map18, #map18, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%162, %99, %79, %163, %59, %96, %164, %76, %56, %161 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%160 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %182 = arith.subf %in_0, %in_1 : f64
      %183 = arith.mulf %in, %182 : f64
      %184 = arith.subf %in_3, %in_4 : f64
      %185 = arith.mulf %in_2, %184 : f64
      %186 = arith.addf %183, %185 : f64
      %187 = arith.subf %in_6, %in_7 : f64
      %188 = arith.mulf %in_5, %187 : f64
      %189 = arith.addf %186, %188 : f64
      %190 = arith.mulf %189, %in_8 : f64
      %191 = arith.addf %out, %190 : f64
      linalg.yield %191 : f64
    } -> tensor<2x5x5x4xf64>
    %166 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%15 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %167 = polygeist.submap(%5, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %168 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%165, %167 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%166 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x5x4x4xf64>
    %169 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%9 : tensor<2x4x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x4x4xf64>
    %170 = polygeist.submap(%6, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %171 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%168, %170 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%169 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<2x4x4x4xf64>
    %172 = polygeist.submap(%0, %c2, %c4, %c4, %c3) {map = #map24} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %173 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%111, %123 : tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>) outs(%172 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.subf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?xf64>
    %174 = polygeist.submapInverse(%0, %173, %c2, %c4, %c4, %c3) {map = #map24} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %175 = polygeist.submap(%174, %c2, %c4, %c3, %c4) {map = #map25} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %176 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%135, %147 : tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>) outs(%175 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.subf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?xf64>
    %177 = polygeist.submapInverse(%174, %176, %c2, %c4, %c3, %c4) {map = #map25} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %178 = polygeist.submap(%177, %c2, %c3, %c4, %c4) {map = #map26} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %179 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%159, %171 : tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>) outs(%178 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.subf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?xf64>
    %180 = polygeist.submapInverse(%177, %179, %c2, %c3, %c4, %c4) {map = #map26} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %181 = bufferization.to_memref %180 : memref<?xf64>
    memref.copy %181, %arg8 : memref<?xf64> to memref<?xf64>
    return
  }
}
