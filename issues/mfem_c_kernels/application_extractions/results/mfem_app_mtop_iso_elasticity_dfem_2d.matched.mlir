#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4 + 32)>
#map10 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 100)>
#map11 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 125)>
#map12 = affine_map<(d0, d1) -> (d1 + d0 * 100)>
#map13 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 25)>
#map14 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 50)>
#map15 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 75)>
#map16 = affine_map<(d0, d1) -> (d1 + d0 * 25)>
#map17 = affine_map<(d0, d1) -> (d0, d1)>
#map18 = affine_map<(d0, d1) -> (d1)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map23 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
#map24 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 100)>
#map25 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 125)>
#map26 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4 + 32)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_mtop_iso_elasticity_dfem_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c25 = arith.constant 25 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg7 : memref<?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?xf64>
    %2 = bufferization.to_tensor %arg5 : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 : memref<?xf64>
    %4 = bufferization.to_tensor %arg3 : memref<?xf64>
    %5 = bufferization.to_tensor %arg2 : memref<?xf64>
    %6 = bufferization.to_tensor %arg1 : memref<?xf64>
    %7 = bufferization.to_tensor %arg0 : memref<?xf64>
    %8 = tensor.empty() : tensor<200xf64>
    %9 = tensor.empty() : tensor<200xf64>
    %10 = tensor.empty() : tensor<2x4x5xf64>
    %11 = tensor.empty() : tensor<2x4x5xf64>
    %extracted_slice = tensor.extract_slice %11[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %12 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice = tensor.insert_slice %12 into %11[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %13 = polygeist.submap(%7, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %14 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v14_contract_15_tc0 = tensor.cast %14 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v13_contract_15_tc1 = tensor.cast %13 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_contract_15_tc2 = tensor.cast %inserted_slice : tensor<2x4x5xf64> to tensor<*xf64>

    %v15_tdyn = kernel.launch @cutensornetContraction2_f64(%v14_contract_15_tc0, %v13_contract_15_tc1, %inserted_slice_contract_15_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %15 = tensor.cast %v15_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %extracted_slice_0 = tensor.extract_slice %10[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %16 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_0 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_1 = tensor.insert_slice %16 into %10[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %17 = polygeist.submap(%6, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %18 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v18_contract_19_tc0 = tensor.cast %18 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v17_contract_19_tc1 = tensor.cast %17 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_1_contract_19_tc2 = tensor.cast %inserted_slice_1 : tensor<2x4x5xf64> to tensor<*xf64>

    %v19_tdyn = kernel.launch @cutensornetContraction2_f64(%v18_contract_19_tc0, %v17_contract_19_tc1, %inserted_slice_1_contract_19_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %19 = tensor.cast %v19_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %20 = polygeist.submap(%9, %c2, %c5, %c5) {map = #map5} : (tensor<200xf64>, index, index, index) -> tensor<?x?x?xf64>
    %21 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%20 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %22 = polygeist.submapInverse(%9, %21, %c2, %c5, %c5) {map = #map5} : (tensor<200xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<200xf64>
    %23 = polygeist.submap(%22, %c2, %c5, %c5) {map = #map5} : (tensor<200xf64>, index, index, index) -> tensor<2x5x5xf64>
    %extracted_slice_2 = tensor.extract_slice %19[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %24 = polygeist.submap(%7, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_2_contract_25_tc0 = tensor.cast %extracted_slice_2 : tensor<?x?x?xf64> to tensor<*xf64>

    %v24_contract_25_tc1 = tensor.cast %24 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v23_contract_25_tc2 = tensor.cast %23 : tensor<2x5x5xf64> to tensor<*xf64>

    %v25_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_2_contract_25_tc0, %v24_contract_25_tc1, %v23_contract_25_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %25 = tensor.cast %v25_tdyn : tensor<*xf64> to tensor<2x5x5xf64>
    %26 = polygeist.submapInverse(%22, %25, %c2, %c5, %c5) {map = #map5} : (tensor<200xf64>, tensor<2x5x5xf64>, index, index, index) -> tensor<200xf64>
    %27 = polygeist.submap(%26, %c2, %c5, %c5) {map = #map8} : (tensor<200xf64>, index, index, index) -> tensor<?x?x?xf64>
    %28 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%27 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %29 = polygeist.submapInverse(%26, %28, %c2, %c5, %c5) {map = #map8} : (tensor<200xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<200xf64>
    %30 = polygeist.submap(%29, %c2, %c5, %c5) {map = #map8} : (tensor<200xf64>, index, index, index) -> tensor<2x5x5xf64>
    %extracted_slice_3 = tensor.extract_slice %15[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %31 = polygeist.submap(%6, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_3_contract_32_tc0 = tensor.cast %extracted_slice_3 : tensor<?x?x?xf64> to tensor<*xf64>

    %v31_contract_32_tc1 = tensor.cast %31 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v30_contract_32_tc2 = tensor.cast %30 : tensor<2x5x5xf64> to tensor<*xf64>

    %v32_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_3_contract_32_tc0, %v31_contract_32_tc1, %v30_contract_32_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %32 = tensor.cast %v32_tdyn : tensor<*xf64> to tensor<2x5x5xf64>
    %33 = polygeist.submapInverse(%29, %32, %c2, %c5, %c5) {map = #map8} : (tensor<200xf64>, tensor<2x5x5xf64>, index, index, index) -> tensor<200xf64>
    %34 = tensor.empty() : tensor<2x4x5xf64>
    %35 = tensor.empty() : tensor<2x4x5xf64>
    %extracted_slice_4 = tensor.extract_slice %35[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %36 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_4 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_5 = tensor.insert_slice %36 into %35[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %37 = polygeist.submap(%7, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %38 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v38_contract_39_tc0 = tensor.cast %38 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v37_contract_39_tc1 = tensor.cast %37 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_5_contract_39_tc2 = tensor.cast %inserted_slice_5 : tensor<2x4x5xf64> to tensor<*xf64>

    %v39_tdyn = kernel.launch @cutensornetContraction2_f64(%v38_contract_39_tc0, %v37_contract_39_tc1, %inserted_slice_5_contract_39_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %39 = tensor.cast %v39_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %extracted_slice_6 = tensor.extract_slice %34[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %40 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_6 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_7 = tensor.insert_slice %40 into %34[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x4x5xf64>
    %41 = polygeist.submap(%6, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v42_contract_43_tc0 = tensor.cast %42 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v41_contract_43_tc1 = tensor.cast %41 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_7_contract_43_tc2 = tensor.cast %inserted_slice_7 : tensor<2x4x5xf64> to tensor<*xf64>

    %v43_tdyn = kernel.launch @cutensornetContraction2_f64(%v42_contract_43_tc0, %v41_contract_43_tc1, %inserted_slice_7_contract_43_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %43 = tensor.cast %v43_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %44 = polygeist.submap(%33, %c2, %c5, %c5) {map = #map10} : (tensor<200xf64>, index, index, index) -> tensor<?x?x?xf64>
    %45 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%44 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %46 = polygeist.submapInverse(%33, %45, %c2, %c5, %c5) {map = #map10} : (tensor<200xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<200xf64>
    %47 = polygeist.submap(%46, %c2, %c5, %c5) {map = #map10} : (tensor<200xf64>, index, index, index) -> tensor<2x5x5xf64>
    %extracted_slice_8 = tensor.extract_slice %43[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %48 = polygeist.submap(%7, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_8_contract_49_tc0 = tensor.cast %extracted_slice_8 : tensor<?x?x?xf64> to tensor<*xf64>

    %v48_contract_49_tc1 = tensor.cast %48 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v47_contract_49_tc2 = tensor.cast %47 : tensor<2x5x5xf64> to tensor<*xf64>

    %v49_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_8_contract_49_tc0, %v48_contract_49_tc1, %v47_contract_49_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %49 = tensor.cast %v49_tdyn : tensor<*xf64> to tensor<2x5x5xf64>
    %50 = polygeist.submapInverse(%46, %49, %c2, %c5, %c5) {map = #map10} : (tensor<200xf64>, tensor<2x5x5xf64>, index, index, index) -> tensor<200xf64>
    %51 = polygeist.submap(%50, %c2, %c5, %c5) {map = #map11} : (tensor<200xf64>, index, index, index) -> tensor<?x?x?xf64>
    %52 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%51 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %53 = polygeist.submapInverse(%50, %52, %c2, %c5, %c5) {map = #map11} : (tensor<200xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<200xf64>
    %54 = polygeist.submap(%53, %c2, %c5, %c5) {map = #map11} : (tensor<200xf64>, index, index, index) -> tensor<2x5x5xf64>
    %extracted_slice_9 = tensor.extract_slice %39[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : tensor<2x4x5xf64> to tensor<?x?x?xf64>
    %55 = polygeist.submap(%6, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_9_contract_56_tc0 = tensor.cast %extracted_slice_9 : tensor<?x?x?xf64> to tensor<*xf64>

    %v55_contract_56_tc1 = tensor.cast %55 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v54_contract_56_tc2 = tensor.cast %54 : tensor<2x5x5xf64> to tensor<*xf64>

    %v56_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_9_contract_56_tc0, %v55_contract_56_tc1, %v54_contract_56_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %56 = tensor.cast %v56_tdyn : tensor<*xf64> to tensor<2x5x5xf64>
    %57 = polygeist.submapInverse(%53, %56, %c2, %c5, %c5) {map = #map11} : (tensor<200xf64>, tensor<2x5x5xf64>, index, index, index) -> tensor<200xf64>
    %58 = polygeist.submap(%8, %c2, %c25) {map = #map12} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %59 = polygeist.submap(%57, %c2, %c25) {map = #map12} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %60 = polygeist.submap(%57, %c2, %c25) {map = #map13} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %61 = polygeist.submap(%57, %c2, %c25) {map = #map14} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %62 = polygeist.submap(%57, %c2, %c25) {map = #map15} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %63 = polygeist.submap(%4, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %64 = polygeist.submap(%3, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %65 = polygeist.submap(%2, %c2, %c25) {map = #map12} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %66 = polygeist.submap(%2, %c2, %c25) {map = #map13} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %67 = polygeist.submap(%2, %c2, %c25) {map = #map14} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %68 = polygeist.submap(%2, %c2, %c25) {map = #map15} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %extracted_slice_10 = tensor.extract_slice %1[0] [%c25] [1] : tensor<?xf64> to tensor<?xf64>
    %69 = linalg.generic {doc = "", indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%65, %66, %67, %68, %59, %60, %61, %62, %extracted_slice_10, %63, %64 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%58 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_30: f64, %in_31: f64, %in_32: f64, %in_33: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %out: f64):
      %153 = arith.mulf %in, %in_32 : f64
      %154 = arith.mulf %in_30, %in_31 : f64
      %155 = arith.subf %153, %154 : f64
      %156 = arith.divf %in_32, %155 : f64
      %157 = arith.negf %in_30 : f64
      %158 = arith.divf %157, %155 : f64
      %159 = arith.addf %in_33, %in_36 : f64
      %160 = arith.mulf %in_37, %155 : f64
      %161 = arith.mulf %in_38, %156 : f64
      %162 = arith.mulf %161, %159 : f64
      %163 = arith.addf %in_33, %in_33 : f64
      %164 = arith.mulf %156, %163 : f64
      %165 = arith.addf %in_34, %in_35 : f64
      %166 = arith.mulf %158, %165 : f64
      %167 = arith.addf %164, %166 : f64
      %168 = arith.mulf %in_39, %167 : f64
      %169 = arith.addf %162, %168 : f64
      %170 = arith.mulf %160, %169 : f64
      linalg.yield %170 : f64
    } -> tensor<?x?xf64>
    %70 = polygeist.submapInverse(%8, %69, %c2, %c25) {map = #map12} : (tensor<200xf64>, tensor<?x?xf64>, index, index) -> tensor<200xf64>
    %71 = polygeist.submap(%70, %c2, %c25) {map = #map14} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %72 = polygeist.submap(%57, %c2, %c25) {map = #map12} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %73 = polygeist.submap(%57, %c2, %c25) {map = #map13} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %74 = polygeist.submap(%57, %c2, %c25) {map = #map14} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %75 = polygeist.submap(%57, %c2, %c25) {map = #map15} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %76 = polygeist.submap(%4, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %77 = polygeist.submap(%3, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %78 = polygeist.submap(%2, %c2, %c25) {map = #map12} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %79 = polygeist.submap(%2, %c2, %c25) {map = #map13} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %80 = polygeist.submap(%2, %c2, %c25) {map = #map14} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %81 = polygeist.submap(%2, %c2, %c25) {map = #map15} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %extracted_slice_11 = tensor.extract_slice %1[0] [%c25] [1] : tensor<?xf64> to tensor<?xf64>
    %82 = linalg.generic {doc = "", indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%78, %79, %80, %81, %72, %73, %74, %75, %extracted_slice_11, %76, %77 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%71 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_30: f64, %in_31: f64, %in_32: f64, %in_33: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %out: f64):
      %153 = arith.mulf %in, %in_32 : f64
      %154 = arith.mulf %in_30, %in_31 : f64
      %155 = arith.subf %153, %154 : f64
      %156 = arith.divf %in_32, %155 : f64
      %157 = arith.negf %in_30 : f64
      %158 = arith.divf %157, %155 : f64
      %159 = arith.addf %in_33, %in_36 : f64
      %160 = arith.mulf %in_37, %155 : f64
      %161 = arith.mulf %in_38, %158 : f64
      %162 = arith.mulf %161, %159 : f64
      %163 = arith.addf %in_35, %in_34 : f64
      %164 = arith.mulf %156, %163 : f64
      %165 = arith.addf %in_36, %in_36 : f64
      %166 = arith.mulf %158, %165 : f64
      %167 = arith.addf %164, %166 : f64
      %168 = arith.mulf %in_39, %167 : f64
      %169 = arith.addf %162, %168 : f64
      %170 = arith.mulf %160, %169 : f64
      linalg.yield %170 : f64
    } -> tensor<?x?xf64>
    %83 = polygeist.submapInverse(%70, %82, %c2, %c25) {map = #map14} : (tensor<200xf64>, tensor<?x?xf64>, index, index) -> tensor<200xf64>
    %84 = polygeist.submap(%83, %c2, %c25) {map = #map13} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %85 = polygeist.submap(%57, %c2, %c25) {map = #map12} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %86 = polygeist.submap(%57, %c2, %c25) {map = #map13} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %87 = polygeist.submap(%57, %c2, %c25) {map = #map14} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %88 = polygeist.submap(%57, %c2, %c25) {map = #map15} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %89 = polygeist.submap(%4, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %90 = polygeist.submap(%3, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %91 = polygeist.submap(%2, %c2, %c25) {map = #map12} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %92 = polygeist.submap(%2, %c2, %c25) {map = #map13} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %93 = polygeist.submap(%2, %c2, %c25) {map = #map14} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %94 = polygeist.submap(%2, %c2, %c25) {map = #map15} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %extracted_slice_12 = tensor.extract_slice %1[0] [%c25] [1] : tensor<?xf64> to tensor<?xf64>
    %95 = linalg.generic {doc = "", indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%91, %92, %93, %94, %85, %86, %87, %88, %extracted_slice_12, %89, %90 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%84 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_30: f64, %in_31: f64, %in_32: f64, %in_33: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %out: f64):
      %153 = arith.mulf %in, %in_32 : f64
      %154 = arith.mulf %in_30, %in_31 : f64
      %155 = arith.subf %153, %154 : f64
      %156 = arith.negf %in_31 : f64
      %157 = arith.divf %156, %155 : f64
      %158 = arith.divf %in, %155 : f64
      %159 = arith.addf %in_33, %in_36 : f64
      %160 = arith.mulf %in_37, %155 : f64
      %161 = arith.mulf %in_38, %157 : f64
      %162 = arith.mulf %161, %159 : f64
      %163 = arith.addf %in_33, %in_33 : f64
      %164 = arith.mulf %157, %163 : f64
      %165 = arith.addf %in_34, %in_35 : f64
      %166 = arith.mulf %158, %165 : f64
      %167 = arith.addf %164, %166 : f64
      %168 = arith.mulf %in_39, %167 : f64
      %169 = arith.addf %162, %168 : f64
      %170 = arith.mulf %160, %169 : f64
      linalg.yield %170 : f64
    } -> tensor<?x?xf64>
    %96 = polygeist.submapInverse(%83, %95, %c2, %c25) {map = #map13} : (tensor<200xf64>, tensor<?x?xf64>, index, index) -> tensor<200xf64>
    %97 = polygeist.submap(%96, %c2, %c25) {map = #map15} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %98 = polygeist.submap(%57, %c2, %c25) {map = #map12} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %99 = polygeist.submap(%57, %c2, %c25) {map = #map13} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %100 = polygeist.submap(%57, %c2, %c25) {map = #map14} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %101 = polygeist.submap(%57, %c2, %c25) {map = #map15} : (tensor<200xf64>, index, index) -> tensor<?x?xf64>
    %102 = polygeist.submap(%4, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %103 = polygeist.submap(%3, %c2, %c25) {map = #map16} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %104 = polygeist.submap(%2, %c2, %c25) {map = #map12} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %105 = polygeist.submap(%2, %c2, %c25) {map = #map13} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %106 = polygeist.submap(%2, %c2, %c25) {map = #map14} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %107 = polygeist.submap(%2, %c2, %c25) {map = #map15} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %extracted_slice_13 = tensor.extract_slice %1[0] [%c25] [1] : tensor<?xf64> to tensor<?xf64>
    %108 = linalg.generic {doc = "", indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%104, %105, %106, %107, %98, %99, %100, %101, %extracted_slice_13, %102, %103 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%97 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_30: f64, %in_31: f64, %in_32: f64, %in_33: f64, %in_34: f64, %in_35: f64, %in_36: f64, %in_37: f64, %in_38: f64, %in_39: f64, %out: f64):
      %153 = arith.mulf %in, %in_32 : f64
      %154 = arith.mulf %in_30, %in_31 : f64
      %155 = arith.subf %153, %154 : f64
      %156 = arith.negf %in_31 : f64
      %157 = arith.divf %156, %155 : f64
      %158 = arith.divf %in, %155 : f64
      %159 = arith.addf %in_33, %in_36 : f64
      %160 = arith.mulf %in_37, %155 : f64
      %161 = arith.mulf %in_38, %158 : f64
      %162 = arith.mulf %161, %159 : f64
      %163 = arith.addf %in_35, %in_34 : f64
      %164 = arith.mulf %157, %163 : f64
      %165 = arith.addf %in_36, %in_36 : f64
      %166 = arith.mulf %158, %165 : f64
      %167 = arith.addf %164, %166 : f64
      %168 = arith.mulf %in_39, %167 : f64
      %169 = arith.addf %162, %168 : f64
      %170 = arith.mulf %160, %169 : f64
      linalg.yield %170 : f64
    } -> tensor<?x?xf64>
    %109 = polygeist.submapInverse(%96, %108, %c2, %c25) {map = #map15} : (tensor<200xf64>, tensor<?x?xf64>, index, index) -> tensor<200xf64>
    %110 = tensor.empty() : tensor<2x4x4xf64>
    %111 = tensor.empty() : tensor<2x4x4xf64>
    %112 = tensor.empty() : tensor<2x5x4xf64>
    %113 = tensor.empty() : tensor<2x5x4xf64>
    %extracted_slice_14 = tensor.extract_slice %113[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %114 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_14 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_15 = tensor.insert_slice %114 into %113[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x5x4xf64>
    %115 = polygeist.submap(%109, %c2, %c5, %c4, %c5) {map = #map19} : (tensor<200xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %116 = polygeist.submap(%6, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v115_contract_117_tc0 = tensor.cast %115 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v116_contract_117_tc1 = tensor.cast %116 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_15_contract_117_tc2 = tensor.cast %inserted_slice_15 : tensor<2x5x4xf64> to tensor<*xf64>

    %v117_tdyn = kernel.launch @cutensornetContraction2_f64(%v115_contract_117_tc0, %v116_contract_117_tc1, %inserted_slice_15_contract_117_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %117 = tensor.cast %v117_tdyn : tensor<*xf64> to tensor<2x5x4xf64>
    %extracted_slice_16 = tensor.extract_slice %112[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %118 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_16 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_17 = tensor.insert_slice %118 into %112[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x5x4xf64>
    %119 = polygeist.submap(%109, %c2, %c5, %c4, %c5) {map = #map21} : (tensor<200xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %120 = polygeist.submap(%7, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v119_contract_121_tc0 = tensor.cast %119 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v120_contract_121_tc1 = tensor.cast %120 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_17_contract_121_tc2 = tensor.cast %inserted_slice_17 : tensor<2x5x4xf64> to tensor<*xf64>

    %v121_tdyn = kernel.launch @cutensornetContraction2_f64(%v119_contract_121_tc0, %v120_contract_121_tc1, %inserted_slice_17_contract_121_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %121 = tensor.cast %v121_tdyn : tensor<*xf64> to tensor<2x5x4xf64>
    %extracted_slice_18 = tensor.extract_slice %111[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : tensor<2x4x4xf64> to tensor<?x?x?xf64>
    %extracted_slice_19 = tensor.extract_slice %117[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %123 = polygeist.submap(%7, %c2, %c4, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_19_contract_124_tc0 = tensor.cast %extracted_slice_19 : tensor<?x?x?xf64> to tensor<*xf64>

    %v123_contract_124_tc1 = tensor.cast %123 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %extracted_slice_18_contract_124_tc2 = tensor.cast %extracted_slice_18 : tensor<?x?x?xf64> to tensor<*xf64>

    %v124_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_19_contract_124_tc0, %v123_contract_124_tc1, %extracted_slice_18_contract_124_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %124 = tensor.cast %v124_tdyn : tensor<*xf64> to tensor<?x?x?xf64>
    %extracted_slice_20 = tensor.extract_slice %110[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : tensor<2x4x4xf64> to tensor<?x?x?xf64>
    %extracted_slice_21 = tensor.extract_slice %121[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %126 = polygeist.submap(%6, %c2, %c4, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_21_contract_127_tc0 = tensor.cast %extracted_slice_21 : tensor<?x?x?xf64> to tensor<*xf64>

    %v126_contract_127_tc1 = tensor.cast %126 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %extracted_slice_20_contract_127_tc2 = tensor.cast %extracted_slice_20 : tensor<?x?x?xf64> to tensor<*xf64>

    %v127_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_21_contract_127_tc0, %v126_contract_127_tc1, %extracted_slice_20_contract_127_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %127 = tensor.cast %v127_tdyn : tensor<*xf64> to tensor<?x?x?xf64>
    %128 = polygeist.submap(%0, %c2, %c4, %c4) {map = #map23} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %129 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%124, %127 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%128 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_30: f64, %out: f64):
      %153 = arith.addf %in, %in_30 : f64
      %154 = arith.addf %out, %153 : f64
      linalg.yield %154 : f64
    } -> tensor<?x?x?xf64>
    %130 = polygeist.submapInverse(%0, %129, %c2, %c4, %c4) {map = #map23} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %131 = tensor.empty() : tensor<2x4x4xf64>
    %132 = tensor.empty() : tensor<2x4x4xf64>
    %133 = tensor.empty() : tensor<2x5x4xf64>
    %134 = tensor.empty() : tensor<2x5x4xf64>
    %extracted_slice_22 = tensor.extract_slice %134[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %135 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_22 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_23 = tensor.insert_slice %135 into %134[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x5x4xf64>
    %136 = polygeist.submap(%109, %c2, %c5, %c4, %c5) {map = #map24} : (tensor<200xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %137 = polygeist.submap(%6, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v136_contract_138_tc0 = tensor.cast %136 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v137_contract_138_tc1 = tensor.cast %137 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_23_contract_138_tc2 = tensor.cast %inserted_slice_23 : tensor<2x5x4xf64> to tensor<*xf64>

    %v138_tdyn = kernel.launch @cutensornetContraction2_f64(%v136_contract_138_tc0, %v137_contract_138_tc1, %inserted_slice_23_contract_138_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %138 = tensor.cast %v138_tdyn : tensor<*xf64> to tensor<2x5x4xf64>
    %extracted_slice_24 = tensor.extract_slice %133[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %139 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_24 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %inserted_slice_25 = tensor.insert_slice %139 into %133[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<?x?x?xf64> into tensor<2x5x4xf64>
    %140 = polygeist.submap(%109, %c2, %c5, %c4, %c5) {map = #map25} : (tensor<200xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %141 = polygeist.submap(%7, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v140_contract_142_tc0 = tensor.cast %140 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v141_contract_142_tc1 = tensor.cast %141 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %inserted_slice_25_contract_142_tc2 = tensor.cast %inserted_slice_25 : tensor<2x5x4xf64> to tensor<*xf64>

    %v142_tdyn = kernel.launch @cutensornetContraction2_f64(%v140_contract_142_tc0, %v141_contract_142_tc1, %inserted_slice_25_contract_142_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %142 = tensor.cast %v142_tdyn : tensor<*xf64> to tensor<2x5x4xf64>
    %extracted_slice_26 = tensor.extract_slice %132[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : tensor<2x4x4xf64> to tensor<?x?x?xf64>
    %extracted_slice_27 = tensor.extract_slice %138[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %144 = polygeist.submap(%7, %c2, %c4, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_27_contract_145_tc0 = tensor.cast %extracted_slice_27 : tensor<?x?x?xf64> to tensor<*xf64>

    %v144_contract_145_tc1 = tensor.cast %144 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %extracted_slice_26_contract_145_tc2 = tensor.cast %extracted_slice_26 : tensor<?x?x?xf64> to tensor<*xf64>

    %v145_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_27_contract_145_tc0, %v144_contract_145_tc1, %extracted_slice_26_contract_145_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %145 = tensor.cast %v145_tdyn : tensor<*xf64> to tensor<?x?x?xf64>
    %extracted_slice_28 = tensor.extract_slice %131[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : tensor<2x4x4xf64> to tensor<?x?x?xf64>
    %extracted_slice_29 = tensor.extract_slice %142[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : tensor<2x5x4xf64> to tensor<?x?x?xf64>
    %147 = polygeist.submap(%6, %c2, %c4, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %extracted_slice_29_contract_148_tc0 = tensor.cast %extracted_slice_29 : tensor<?x?x?xf64> to tensor<*xf64>

    %v147_contract_148_tc1 = tensor.cast %147 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %extracted_slice_28_contract_148_tc2 = tensor.cast %extracted_slice_28 : tensor<?x?x?xf64> to tensor<*xf64>

    %v148_tdyn = kernel.launch @cutensornetContraction2_f64(%extracted_slice_29_contract_148_tc0, %v147_contract_148_tc1, %extracted_slice_28_contract_148_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %148 = tensor.cast %v148_tdyn : tensor<*xf64> to tensor<?x?x?xf64>
    %149 = polygeist.submap(%130, %c2, %c4, %c4) {map = #map26} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %150 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%145, %148 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%149 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_30: f64, %out: f64):
      %153 = arith.addf %in, %in_30 : f64
      %154 = arith.addf %out, %153 : f64
      linalg.yield %154 : f64
    } -> tensor<?x?x?xf64>
    %151 = polygeist.submapInverse(%130, %150, %c2, %c4, %c4) {map = #map26} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %152 = bufferization.to_memref %151 : memref<?xf64>
    memref.copy %152, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
}

