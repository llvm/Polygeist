#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d1 + d0 * 375 + d2 * 5)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d1 + d0 * 375 + d2 * 5 + 125)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d1 + d0 * 375 + d2 * 5 + 250)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d2)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d1)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_grad_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg1 : memref<?xf64>
    %3 = bufferization.to_tensor %arg0 : memref<?xf64>
    %4 = tensor.empty() : tensor<2x4x4x4xf64>
    %5 = tensor.empty() : tensor<2x4x4x4xf64>
    %6 = tensor.empty() : tensor<2x4x4x4xf64>
    %7 = tensor.empty() : tensor<2x5x4x4xf64>
    %8 = tensor.empty() : tensor<2x5x4x4xf64>
    %9 = tensor.empty() : tensor<2x5x4x4xf64>
    %10 = tensor.empty() : tensor<2x5x5x4xf64>
    %11 = tensor.empty() : tensor<2x5x5x4xf64>
    %12 = tensor.empty() : tensor<2x5x5x4xf64>
    %14 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %15 = polygeist.submap(%1, %c2, %c5, %c5, %c4, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v12_tc2 = tensor.cast %12 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v16_tdyn = kernel.launch @cublasGemmFor1x1Conv(%14, %15, %v12_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %16 = tensor.cast %v16_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x4xf64>
    %18 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %19 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v11_tc2 = tensor.cast %11 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v20_tdyn = kernel.launch @cublasGemmFor1x1Conv(%18, %19, %v11_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %20 = tensor.cast %v20_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x4xf64>
    %22 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %23 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v10_tc2 = tensor.cast %10 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v24_tdyn = kernel.launch @cublasGemmFor1x1Conv(%22, %23, %v10_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %24 = tensor.cast %v24_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x4xf64>
    %26 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v16_tc1 = tensor.cast %16 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v9_tc2 = tensor.cast %9 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v27_tdyn = kernel.launch @cublasGemmFor1x1Conv(%26, %v16_tc1, %v9_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %27 = tensor.cast %v27_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %29 = polygeist.submap(%1, %c2, %c5, %c4, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v20_tc1 = tensor.cast %20 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v8_tc2 = tensor.cast %8 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v30_tdyn = kernel.launch @cublasGemmFor1x1Conv(%29, %v20_tc1, %v8_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %30 = tensor.cast %v30_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %32 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v24_tc1 = tensor.cast %24 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v7_tc2 = tensor.cast %7 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v33_tdyn = kernel.launch @cublasGemmFor1x1Conv(%32, %v24_tc1, %v7_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %33 = tensor.cast %v33_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %35 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v27_tc1 = tensor.cast %27 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v6_tc2 = tensor.cast %6 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v36_tdyn = kernel.launch @cublasGemmFor1x1Conv(%35, %v27_tc1, %v6_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %36 = tensor.cast %v36_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %38 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v30_tc1 = tensor.cast %30 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v5_tc2 = tensor.cast %5 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v39_tdyn = kernel.launch @cublasGemmFor1x1Conv(%38, %v30_tc1, %v5_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %39 = tensor.cast %v39_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %41 = polygeist.submap(%1, %c2, %c4, %c4, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v33_tc1 = tensor.cast %33 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v4_tc2 = tensor.cast %4 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v42_tdyn = kernel.launch @cublasGemmFor1x1Conv(%41, %v33_tc1, %v4_tc2) : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %42 = tensor.cast %v42_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %43 = polygeist.submap(%0, %c2, %c4, %c4, %c4) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %44 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%36, %39, %42 : tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>) outs(%43 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %47 = arith.addf %in, %in_0 : f64
      %48 = arith.addf %47, %in_1 : f64
      %49 = arith.addf %out, %48 : f64
      linalg.yield %49 : f64
    } -> tensor<?x?x?x?xf64>
    %45 = polygeist.submapInverse(%0, %44, %c2, %c4, %c4, %c4) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %46 = bufferization.to_memref %45 : memref<?xf64>
    memref.copy %46, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}
