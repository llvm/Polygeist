#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map11 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
#map12 = affine_map<(d0, d1) -> (d0 + d1 * 5 + 25)>
#map13 = affine_map<(d0, d1) -> (d0 * 4 + d1 * 20)>
#map14 = affine_map<(d0, d1) -> (d0 * 4 + d1 * 20 + 1)>
#map15 = affine_map<(d0, d1) -> (d0 * 4 + d1 * 20 + 2)>
#map16 = affine_map<(d0, d1) -> (d0 * 4 + d1 * 20 + 3)>
#map17 = affine_map<(d0, d1) -> (d0 + d1 * 5 + 50)>
#map18 = affine_map<(d0, d1) -> (d0 + d1 * 5 + 75)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map23 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_dfem_minimal_surface_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg5 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = bufferization.to_tensor %arg2 : memref<?xf64>
    %4 = bufferization.to_tensor %arg1 : memref<?xf64>
    %5 = bufferization.to_tensor %arg0 : memref<?xf64>
    %6 = tensor.empty() : tensor<100xf64>
    %7 = tensor.empty() : tensor<100xf64>
    %8 = tensor.empty() : tensor<2x4x5xf64>
    %9 = tensor.empty() : tensor<2x4x5xf64>
    %11 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %12 = polygeist.submap(%3, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v12_contract_13_tc0 = tensor.cast %12 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v11_contract_13_tc1 = tensor.cast %11 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v9_contract_13_tc2 = tensor.cast %9 : tensor<2x4x5xf64> to tensor<*xf64>

    %v13_tdyn = kernel.launch @cutensornetContraction2_f64(%v12_contract_13_tc0, %v11_contract_13_tc1, %v9_contract_13_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %13 = tensor.cast %v13_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %15 = polygeist.submap(%4, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %16 = polygeist.submap(%3, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v16_contract_17_tc0 = tensor.cast %16 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v15_contract_17_tc1 = tensor.cast %15 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v8_contract_17_tc2 = tensor.cast %8 : tensor<2x4x5xf64> to tensor<*xf64>

    %v17_tdyn = kernel.launch @cutensornetContraction2_f64(%v16_contract_17_tc0, %v15_contract_17_tc1, %v8_contract_17_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %17 = tensor.cast %v17_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %18 = polygeist.submap(%7, %c2, %c5, %c5) {map = #map5} : (tensor<100xf64>, index, index, index) -> tensor<?x?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%18 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %20 = polygeist.submapInverse(%7, %19, %c2, %c5, %c5) {map = #map5} : (tensor<100xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<100xf64>
    %21 = polygeist.submap(%20, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = polygeist.submap(%5, %c2, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v17_contract_23_tc0 = tensor.cast %17 : tensor<2x4x5xf64> to tensor<*xf64>

    %v22_contract_23_tc1 = tensor.cast %22 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v21_contract_23_tc2 = tensor.cast %21 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v23_tdyn = kernel.launch @cutensornetContraction2_f64(%v17_contract_23_tc0, %v22_contract_23_tc1, %v21_contract_23_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %23 = tensor.cast %v23_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %24 = polygeist.submapInverse(%20, %23, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<100xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<100xf64>
    %25 = polygeist.submap(%24, %c2, %c5, %c5) {map = #map9} : (tensor<100xf64>, index, index, index) -> tensor<?x?x?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%25 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %27 = polygeist.submapInverse(%24, %26, %c2, %c5, %c5) {map = #map9} : (tensor<100xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<100xf64>
    %28 = polygeist.submap(%27, %c2, %c5, %c5, %c4) {map = #map10} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %29 = polygeist.submap(%4, %c2, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v13_contract_30_tc0 = tensor.cast %13 : tensor<2x4x5xf64> to tensor<*xf64>

    %v29_contract_30_tc1 = tensor.cast %29 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v28_contract_30_tc2 = tensor.cast %28 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v30_tdyn = kernel.launch @cutensornetContraction2_f64(%v13_contract_30_tc0, %v29_contract_30_tc1, %v28_contract_30_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %30 = tensor.cast %v30_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %31 = polygeist.submapInverse(%27, %30, %c2, %c5, %c5, %c4) {map = #map10} : (tensor<100xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<100xf64>
    %32 = affine.for %arg6 = 0 to 5 iter_args(%arg7 = %6) -> (tensor<100xf64>) {
      %55 = affine.for %arg8 = 0 to 5 iter_args(%arg9 = %arg7) -> (tensor<100xf64>) {
        %56 = affine.apply #map11(%arg8, %arg6)
        %extracted = tensor.extract %31[%56] : tensor<100xf64>
        %57 = affine.apply #map12(%arg8, %arg6)
        %extracted_1 = tensor.extract %31[%57] : tensor<100xf64>
        %58 = affine.apply #map13(%arg8, %arg6)
        %extracted_2 = tensor.extract %2[%58] : tensor<?xf64>
        %59 = affine.apply #map14(%arg8, %arg6)
        %extracted_3 = tensor.extract %2[%59] : tensor<?xf64>
        %60 = affine.apply #map15(%arg8, %arg6)
        %extracted_4 = tensor.extract %2[%60] : tensor<?xf64>
        %61 = affine.apply #map16(%arg8, %arg6)
        %extracted_5 = tensor.extract %2[%61] : tensor<?xf64>
        %62 = arith.mulf %extracted_2, %extracted_5 : f64
        %63 = arith.mulf %extracted_3, %extracted_4 : f64
        %64 = arith.subf %62, %63 : f64
        %65 = arith.divf %extracted_5, %64 : f64
        %66 = arith.negf %extracted_3 : f64
        %67 = arith.divf %66, %64 : f64
        %68 = arith.negf %extracted_4 : f64
        %69 = arith.divf %68, %64 : f64
        %70 = arith.divf %extracted_2, %64 : f64
        %71 = arith.mulf %extracted, %65 : f64
        %72 = arith.mulf %extracted_1, %69 : f64
        %73 = arith.addf %71, %72 : f64
        %74 = arith.mulf %extracted, %67 : f64
        %75 = arith.mulf %extracted_1, %70 : f64
        %76 = arith.addf %74, %75 : f64
        %77 = arith.mulf %73, %73 : f64
        %78 = arith.addf %77, %cst : f64
        %79 = arith.mulf %76, %76 : f64
        %80 = arith.addf %78, %79 : f64
        %81 = math.sqrt %80 : f64
        %82 = arith.divf %cst, %81 : f64
        %83 = arith.mulf %82, %64 : f64
        %84 = affine.apply #map11(%arg8, %arg6)
        %extracted_6 = tensor.extract %1[%84] : tensor<?xf64>
        %85 = arith.mulf %83, %extracted_6 : f64
        %86 = arith.mulf %73, %65 : f64
        %87 = arith.mulf %76, %67 : f64
        %88 = arith.addf %86, %87 : f64
        %89 = arith.mulf %85, %88 : f64
        %90 = affine.apply #map11(%arg8, %arg6)
        %inserted = tensor.insert %89 into %arg9[%90] : tensor<100xf64>
        %91 = arith.mulf %73, %69 : f64
        %92 = arith.mulf %76, %70 : f64
        %93 = arith.addf %91, %92 : f64
        %94 = arith.mulf %85, %93 : f64
        %95 = affine.apply #map12(%arg8, %arg6)
        %inserted_7 = tensor.insert %94 into %inserted[%95] : tensor<100xf64>
        %96 = affine.apply #map17(%arg8, %arg6)
        %inserted_8 = tensor.insert %cst_0 into %inserted_7[%96] : tensor<100xf64>
        %97 = affine.apply #map18(%arg8, %arg6)
        %inserted_9 = tensor.insert %cst_0 into %inserted_8[%97] : tensor<100xf64>
        affine.yield %inserted_9 : tensor<100xf64>
      }
      affine.yield %55 : tensor<100xf64>
    }
    %33 = tensor.empty() : tensor<2x4x4xf64>
    %34 = tensor.empty() : tensor<2x4x4xf64>
    %35 = tensor.empty() : tensor<2x5x4xf64>
    %36 = tensor.empty() : tensor<2x5x4xf64>
    %38 = polygeist.submap(%32, %c2, %c5, %c4, %c5) {map = #map19} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %39 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v38_contract_40_tc0 = tensor.cast %38 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v39_contract_40_tc1 = tensor.cast %39 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v36_contract_40_tc2 = tensor.cast %36 : tensor<2x5x4xf64> to tensor<*xf64>

    %v40_tdyn = kernel.launch @cutensornetContraction2_f64(%v38_contract_40_tc0, %v39_contract_40_tc1, %v36_contract_40_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %40 = tensor.cast %v40_tdyn : tensor<*xf64> to tensor<2x5x4xf64>
    %42 = polygeist.submap(%32, %c2, %c5, %c4, %c5) {map = #map21} : (tensor<100xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %43 = polygeist.submap(%5, %c2, %c5, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v42_contract_44_tc0 = tensor.cast %42 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v43_contract_44_tc1 = tensor.cast %43 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v35_contract_44_tc2 = tensor.cast %35 : tensor<2x5x4xf64> to tensor<*xf64>

    %v44_tdyn = kernel.launch @cutensornetContraction2_f64(%v42_contract_44_tc0, %v43_contract_44_tc1, %v35_contract_44_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %44 = tensor.cast %v44_tdyn : tensor<*xf64> to tensor<2x5x4xf64>
    %46 = polygeist.submap(%5, %c2, %c4, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v40_contract_47_tc0 = tensor.cast %40 : tensor<2x5x4xf64> to tensor<*xf64>

    %v46_contract_47_tc1 = tensor.cast %46 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v34_contract_47_tc2 = tensor.cast %34 : tensor<2x4x4xf64> to tensor<*xf64>

    %v47_tdyn = kernel.launch @cutensornetContraction2_f64(%v40_contract_47_tc0, %v46_contract_47_tc1, %v34_contract_47_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %47 = tensor.cast %v47_tdyn : tensor<*xf64> to tensor<2x4x4xf64>
    %49 = polygeist.submap(%4, %c2, %c4, %c4, %c5) {map = #map22} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v44_contract_50_tc0 = tensor.cast %44 : tensor<2x5x4xf64> to tensor<*xf64>

    %v49_contract_50_tc1 = tensor.cast %49 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v33_contract_50_tc2 = tensor.cast %33 : tensor<2x4x4xf64> to tensor<*xf64>

    %v50_tdyn = kernel.launch @cutensornetContraction2_f64(%v44_contract_50_tc0, %v49_contract_50_tc1, %v33_contract_50_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %50 = tensor.cast %v50_tdyn : tensor<*xf64> to tensor<2x4x4xf64>
    %51 = polygeist.submap(%0, %c2, %c4, %c4) {map = #map23} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %52 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = "", polygeist.pointwise_plan = {backend = "cutensor", coverage = "whole", dtype = "f64", ops = 2 : i32, regions = 1 : i32, temps = 1 : i32, version = 1 : i32}} ins(%47, %50 : tensor<2x4x4xf64>, tensor<2x4x4xf64>) outs(%51 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %55 = arith.addf %in, %in_1 : f64
      %56 = arith.addf %out, %55 : f64
      linalg.yield %56 : f64
    } -> tensor<?x?x?xf64>
    %53 = polygeist.submapInverse(%0, %52, %c2, %c4, %c4) {map = #map23} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %54 = bufferization.to_memref %53 : memref<?xf64>
    memref.copy %54, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}
