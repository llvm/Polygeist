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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex35p_hcurl_3d_partial(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %45 = polygeist.submap(%8, %c2, %c4, %c4, %c5, %c3) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c3) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v43_contract_47_tc2 = tensor.cast %43 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v47_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%46, %45, %v43_contract_47_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %47 = tensor.cast %v47_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %49 = polygeist.submap(%4, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v47_contract_50_tc0 = tensor.cast %47 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v38_contract_50_tc2 = tensor.cast %38 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v50_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v47_contract_50_tc0, %49, %v38_contract_50_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %50 = tensor.cast %v50_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %52 = polygeist.submap(%7, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v47_contract_53_tc0 = tensor.cast %47 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v37_contract_53_tc2 = tensor.cast %37 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v53_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v47_contract_53_tc0, %52, %v37_contract_53_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %53 = tensor.cast %v53_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %55 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v50_contract_56_tc0 = tensor.cast %50 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v32_contract_56_tc2 = tensor.cast %32 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v56_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v50_contract_56_tc0, %55, %v32_contract_56_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %56 = tensor.cast %v56_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %58 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v53_contract_59_tc0 = tensor.cast %53 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v31_contract_59_tc2 = tensor.cast %31 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v59_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v53_contract_59_tc0, %58, %v31_contract_59_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %59 = tensor.cast %v59_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %61 = polygeist.submap(%4, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%1, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v42_contract_63_tc2 = tensor.cast %42 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v63_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%62, %61, %v42_contract_63_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %63 = tensor.cast %v63_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %65 = polygeist.submap(%7, %c2, %c4, %c3, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %66 = polygeist.submap(%1, %c2, %c4, %c3, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v41_contract_67_tc2 = tensor.cast %41 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v67_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%66, %65, %v41_contract_67_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %67 = tensor.cast %v67_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %69 = polygeist.submap(%8, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v63_contract_70_tc0 = tensor.cast %63 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v36_contract_70_tc2 = tensor.cast %36 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v70_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v63_contract_70_tc0, %69, %v36_contract_70_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %70 = tensor.cast %v70_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %72 = polygeist.submap(%8, %c2, %c4, %c5, %c5, %c3) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v67_contract_73_tc0 = tensor.cast %67 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v35_contract_73_tc2 = tensor.cast %35 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v73_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v67_contract_73_tc0, %72, %v35_contract_73_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %73 = tensor.cast %v73_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %75 = polygeist.submap(%7, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v70_contract_76_tc0 = tensor.cast %70 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v30_contract_76_tc2 = tensor.cast %30 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v76_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v70_contract_76_tc0, %75, %v30_contract_76_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %76 = tensor.cast %v76_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %78 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v73_contract_79_tc0 = tensor.cast %73 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v29_contract_79_tc2 = tensor.cast %29 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v79_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v73_contract_79_tc0, %78, %v29_contract_79_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %79 = tensor.cast %v79_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %81 = polygeist.submap(%4, %c2, %c3, %c4, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %82 = polygeist.submap(%1, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v40_contract_83_tc2 = tensor.cast %40 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v83_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%82, %81, %v40_contract_83_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %83 = tensor.cast %v83_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %85 = polygeist.submap(%7, %c2, %c3, %c4, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = polygeist.submap(%1, %c2, %c3, %c4, %c5, %c4) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v39_contract_87_tc2 = tensor.cast %39 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v87_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%86, %85, %v39_contract_87_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %87 = tensor.cast %v87_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %89 = polygeist.submap(%7, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v83_contract_90_tc0 = tensor.cast %83 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v34_contract_90_tc2 = tensor.cast %34 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v90_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v83_contract_90_tc0, %89, %v34_contract_90_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %90 = tensor.cast %v90_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %92 = polygeist.submap(%4, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v87_contract_93_tc0 = tensor.cast %87 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v33_contract_93_tc2 = tensor.cast %33 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v93_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v87_contract_93_tc0, %92, %v33_contract_93_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %93 = tensor.cast %v93_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %95 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v90_contract_96_tc0 = tensor.cast %90 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v28_contract_96_tc2 = tensor.cast %28 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v96_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v90_contract_96_tc0, %95, %v28_contract_96_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %96 = tensor.cast %v96_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %98 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c3) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v93_contract_99_tc0 = tensor.cast %93 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v27_contract_99_tc2 = tensor.cast %27 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v99_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v93_contract_99_tc0, %98, %v27_contract_99_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %99 = tensor.cast %v99_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
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
    %107 = polygeist.submap(%5, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v105_contract_108_tc0 = tensor.cast %105 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v20_contract_108_tc2 = tensor.cast %20 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v108_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v105_contract_108_tc0, %107, %v20_contract_108_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %108 = tensor.cast %v108_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %110 = polygeist.submap(%3, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v108_contract_111_tc0 = tensor.cast %108 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v14_contract_111_tc2 = tensor.cast %14 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v111_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v108_contract_111_tc0, %110, %v14_contract_111_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %111 = tensor.cast %v111_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
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
    %119 = polygeist.submap(%3, %c2, %c5, %c4, %c3, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v117_contract_120_tc0 = tensor.cast %117 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v19_contract_120_tc2 = tensor.cast %19 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v120_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v117_contract_120_tc0, %119, %v19_contract_120_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %120 = tensor.cast %v120_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %122 = polygeist.submap(%5, %c2, %c4, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v120_contract_123_tc0 = tensor.cast %120 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v13_contract_123_tc2 = tensor.cast %13 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v123_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v120_contract_123_tc0, %122, %v13_contract_123_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %123 = tensor.cast %v123_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
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
    %131 = polygeist.submap(%6, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v129_contract_132_tc0 = tensor.cast %129 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v18_contract_132_tc2 = tensor.cast %18 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v132_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v129_contract_132_tc0, %131, %v18_contract_132_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %132 = tensor.cast %v132_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %134 = polygeist.submap(%5, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v132_contract_135_tc0 = tensor.cast %132 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v12_contract_135_tc2 = tensor.cast %12 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v135_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v132_contract_135_tc0, %134, %v12_contract_135_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %135 = tensor.cast %v135_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
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
    %143 = polygeist.submap(%6, %c2, %c5, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v141_contract_144_tc0 = tensor.cast %141 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v17_contract_144_tc2 = tensor.cast %17 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v144_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v141_contract_144_tc0, %143, %v17_contract_144_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %144 = tensor.cast %v144_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %146 = polygeist.submap(%3, %c2, %c4, %c3, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v144_contract_147_tc0 = tensor.cast %144 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v11_contract_147_tc2 = tensor.cast %11 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v147_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v144_contract_147_tc0, %146, %v11_contract_147_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %147 = tensor.cast %v147_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
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
    %155 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v153_contract_156_tc0 = tensor.cast %153 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v16_contract_156_tc2 = tensor.cast %16 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v156_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v153_contract_156_tc0, %155, %v16_contract_156_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %156 = tensor.cast %v156_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %158 = polygeist.submap(%6, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v156_contract_159_tc0 = tensor.cast %156 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v10_contract_159_tc2 = tensor.cast %10 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v159_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v156_contract_159_tc0, %158, %v10_contract_159_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %159 = tensor.cast %v159_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
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
    %167 = polygeist.submap(%5, %c2, %c5, %c4, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v165_contract_168_tc0 = tensor.cast %165 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v15_contract_168_tc2 = tensor.cast %15 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v168_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v165_contract_168_tc0, %167, %v15_contract_168_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %168 = tensor.cast %v168_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %170 = polygeist.submap(%6, %c2, %c3, %c4, %c4, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v168_contract_171_tc0 = tensor.cast %168 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v9_contract_171_tc2 = tensor.cast %9 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v171_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v168_contract_171_tc0, %170, %v9_contract_171_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %171 = tensor.cast %v171_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
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
