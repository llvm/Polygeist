#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50)>
#map9 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map11 = affine_map<(d0, d1) -> (d1 + d0 * 5)>
#map12 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 25)>
#map13 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20)>
#map14 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 1)>
#map15 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 2)>
#map16 = affine_map<(d0, d1) -> (d1 * 4 + d0 * 20 + 3)>
#map17 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 50)>
#map18 = affine_map<(d0, d1) -> (d1 + d0 * 5 + 75)>
#map19 = affine_map<(d0, d1) -> (d0, d1)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map23 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map24 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_dfem_minimal_surface_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %6 = tensor.empty() : tensor<100xf64>
    %7 = tensor.empty() : tensor<100xf64>
    %8 = tensor.empty() : tensor<2x4x5xf64>
    %9 = tensor.empty() : tensor<2x4x5xf64>
    %10 = polygeist.submap(%9, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %11 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %12 = polygeist.submapInverse(%9, %11, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %13 = polygeist.submap(%2, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %14 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %15 = polygeist.submap(%12, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v13_contract_16_tc0 = tensor.cast %13 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v14_contract_16_tc1 = tensor.cast %14 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v15_contract_16_tc2 = tensor.cast %15 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v16_tdyn = kernel.launch @cutensornetContraction2_f64(%v13_contract_16_tc0, %v14_contract_16_tc1, %v15_contract_16_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %16 = tensor.cast %v16_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %17 = polygeist.submapInverse(%12, %16, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %18 = polygeist.submap(%8, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%18 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %20 = polygeist.submapInverse(%8, %19, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %21 = polygeist.submap(%2, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %23 = polygeist.submap(%20, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v21_contract_24_tc0 = tensor.cast %21 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v22_contract_24_tc1 = tensor.cast %22 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v23_contract_24_tc2 = tensor.cast %23 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v24_tdyn = kernel.launch @cutensornetContraction2_f64(%v21_contract_24_tc0, %v22_contract_24_tc1, %v23_contract_24_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %24 = tensor.cast %v24_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %25 = polygeist.submapInverse(%20, %24, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %26 = polygeist.submap(%7, %c2, %c5, %c5) {map = #map5} : (tensor<100xf64>, index, index, index) -> tensor<?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%26 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %28 = polygeist.submapInverse(%7, %27, %c2, %c5, %c5) {map = #map5} : (tensor<100xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<100xf64>
    %29 = polygeist.submap(%25, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %30 = polygeist.submap(%0, %c2, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %31 = polygeist.submap(%28, %c2, %c5, %c5, %c4) {map = #map8} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %32 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%29, %30 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%31 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.mulf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?x?xf64>
    %33 = polygeist.submapInverse(%28, %32, %c2, %c5, %c5, %c4) {map = #map8} : (tensor<100xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<100xf64>
    %34 = polygeist.submap(%33, %c2, %c5, %c5) {map = #map9} : (tensor<100xf64>, index, index, index) -> tensor<?x?x?xf64>
    %35 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%34 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %36 = polygeist.submapInverse(%33, %35, %c2, %c5, %c5) {map = #map9} : (tensor<100xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<100xf64>
    %37 = polygeist.submap(%17, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %38 = polygeist.submap(%1, %c2, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %39 = polygeist.submap(%36, %c2, %c5, %c5, %c4) {map = #map10} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %40 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%37, %38 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%39 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.mulf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?x?xf64>
    %41 = polygeist.submapInverse(%36, %40, %c2, %c5, %c5, %c4) {map = #map10} : (tensor<100xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<100xf64>
    %42 = polygeist.submap(%41, %c5, %c5) {map = #map11} : (tensor<100xf64>, index, index) -> tensor<?x?xf64>
    %43 = polygeist.submap(%41, %c5, %c5) {map = #map12} : (tensor<100xf64>, index, index) -> tensor<?x?xf64>
    %44 = polygeist.submap(%3, %c5, %c5) {map = #map13} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %45 = polygeist.submap(%3, %c5, %c5) {map = #map14} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %46 = polygeist.submap(%3, %c5, %c5) {map = #map15} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %47 = polygeist.submap(%3, %c5, %c5) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %48 = polygeist.submap(%4, %c5, %c5) {map = #map11} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %49 = polygeist.submap(%6, %c5, %c5) {map = #map11} : (tensor<100xf64>, index, index) -> tensor<?x?xf64>
    %50 = polygeist.submap(%6, %c5, %c5) {map = #map12} : (tensor<100xf64>, index, index) -> tensor<?x?xf64>
    %51 = polygeist.submap(%6, %c5, %c5) {map = #map17} : (tensor<100xf64>, index, index) -> tensor<?x?xf64>
    %52 = polygeist.submap(%6, %c5, %c5) {map = #map18} : (tensor<100xf64>, index, index) -> tensor<?x?xf64>
    %53:4 = linalg.generic {doc = "", indexing_maps = [#map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19, #map19], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%42, %43, %44, %45, %46, %47, %48 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%49, %50, %51, %52 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64, %out_7: f64, %out_8: f64, %out_9: f64):
      %100 = arith.mulf %in_2, %in_5 : f64
      %101 = arith.mulf %in_3, %in_4 : f64
      %102 = arith.subf %100, %101 : f64
      %103 = arith.divf %in_5, %102 : f64
      %104 = arith.negf %in_3 : f64
      %105 = arith.divf %104, %102 : f64
      %106 = arith.negf %in_4 : f64
      %107 = arith.divf %106, %102 : f64
      %108 = arith.divf %in_2, %102 : f64
      %109 = arith.mulf %in, %103 : f64
      %110 = arith.mulf %in_1, %107 : f64
      %111 = arith.addf %109, %110 : f64
      %112 = arith.mulf %in, %105 : f64
      %113 = arith.mulf %in_1, %108 : f64
      %114 = arith.addf %112, %113 : f64
      %115 = arith.mulf %111, %111 : f64
      %116 = arith.addf %115, %cst : f64
      %117 = arith.mulf %114, %114 : f64
      %118 = arith.addf %116, %117 : f64
      %119 = math.sqrt %118 : f64
      %120 = arith.divf %cst, %119 : f64
      %121 = arith.mulf %120, %102 : f64
      %122 = arith.mulf %121, %in_6 : f64
      %123 = arith.mulf %111, %103 : f64
      %124 = arith.mulf %114, %105 : f64
      %125 = arith.addf %123, %124 : f64
      %126 = arith.mulf %122, %125 : f64
      %127 = arith.mulf %111, %107 : f64
      %128 = arith.mulf %114, %108 : f64
      %129 = arith.addf %127, %128 : f64
      %130 = arith.mulf %122, %129 : f64
      linalg.yield %126, %130, %cst_0, %cst_0 : f64, f64, f64, f64
    } -> (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
    %54 = polygeist.submapInverse(%6, %53#0, %c5, %c5) {map = #map11} : (tensor<100xf64>, tensor<?x?xf64>, index, index) -> tensor<100xf64>
    %55 = polygeist.submapInverse(%54, %53#1, %c5, %c5) {map = #map12} : (tensor<100xf64>, tensor<?x?xf64>, index, index) -> tensor<100xf64>
    %56 = polygeist.submapInverse(%55, %53#2, %c5, %c5) {map = #map17} : (tensor<100xf64>, tensor<?x?xf64>, index, index) -> tensor<100xf64>
    %57 = polygeist.submapInverse(%56, %53#3, %c5, %c5) {map = #map18} : (tensor<100xf64>, tensor<?x?xf64>, index, index) -> tensor<100xf64>
    %58 = tensor.empty() : tensor<2x4x4xf64>
    %59 = tensor.empty() : tensor<2x4x4xf64>
    %60 = tensor.empty() : tensor<2x5x4xf64>
    %61 = tensor.empty() : tensor<2x5x4xf64>
    %62 = polygeist.submap(%61, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%62 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %64 = polygeist.submapInverse(%61, %63, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %65 = polygeist.submap(%57, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %66 = polygeist.submap(%1, %c2, %c5, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %67 = polygeist.submap(%64, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %68 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%65, %66 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%67 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.mulf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?x?xf64>
    %69 = polygeist.submapInverse(%64, %68, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %70 = polygeist.submap(%60, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%70 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %72 = polygeist.submapInverse(%60, %71, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %73 = polygeist.submap(%57, %c2, %c5, %c4, %c5) {map = #map22} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %74 = polygeist.submap(%0, %c2, %c5, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %75 = polygeist.submap(%72, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %76 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%73, %74 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%75 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.mulf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?x?xf64>
    %77 = polygeist.submapInverse(%72, %76, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %78 = polygeist.submap(%59, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %79 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%78 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %80 = polygeist.submapInverse(%59, %79, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %81 = polygeist.submap(%69, %c2, %c4, %c4, %c5) {map = #map6} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %82 = polygeist.submap(%0, %c2, %c4, %c4, %c5) {map = #map23} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %83 = polygeist.submap(%80, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %84 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%81, %82 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%83 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.mulf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?x?xf64>
    %85 = polygeist.submapInverse(%80, %84, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4xf64>
    %86 = polygeist.submap(%58, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %87 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%86 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %88 = polygeist.submapInverse(%58, %87, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %89 = polygeist.submap(%77, %c2, %c4, %c4, %c5) {map = #map6} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %90 = polygeist.submap(%1, %c2, %c4, %c4, %c5) {map = #map23} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %91 = polygeist.submap(%88, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %92 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%89, %90 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%91 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.mulf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?x?xf64>
    %93 = polygeist.submapInverse(%88, %92, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4xf64>
    %94 = polygeist.submap(%85, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %95 = polygeist.submap(%93, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %96 = polygeist.submap(%5, %c2, %c4, %c4) {map = #map24} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %97 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%94, %95 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%96 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %100 = arith.addf %in, %in_1 : f64
      %101 = arith.addf %out, %100 : f64
      linalg.yield %101 : f64
    } -> tensor<?x?x?xf64>
    %98 = polygeist.submapInverse(%5, %97, %c2, %c4, %c4) {map = #map24} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %99 = bufferization.to_memref %98 : memref<?xf64>
    memref.copy %99, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}
