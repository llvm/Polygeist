#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<()[s0] -> (s0 - 1)>
#map5 = affine_map<(d0) -> (d0, d0)>
#map6 = affine_map<(d0) -> (d0 + 1)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map8 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-01 : f64
    %0 = bufferization.to_tensor %arg3 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 : memref<?xf64>
    %3 = bufferization.to_tensor %arg6 : memref<?xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = polygeist.submap(%2, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%6 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?xf64>
    %8 = polygeist.submapInverse(%2, %7, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %9 = polygeist.submap(%0, %5, %4) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %10 = polygeist.submap(%8, %5, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %11 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%9 : tensor<?x?xf64>) outs(%10 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %54 = arith.addf %out, %in : f64
      linalg.yield %54 : f64
    } -> tensor<?x?xf64>
    %12 = polygeist.submapInverse(%8, %11, %5, %4) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %13 = polygeist.submap(%12, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %14 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%13 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %54 = arith.divf %out, %arg2 : f64
      linalg.yield %54 : f64
    } -> tensor<?xf64>
    %15 = polygeist.submapInverse(%12, %14, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %16 = bufferization.to_memref %15 : memref<?xf64>
    memref.copy %16, %arg5 : memref<?xf64> to memref<?xf64>
    %17 = polygeist.submap(%3, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%17 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?xf64>
    %19 = polygeist.submapInverse(%3, %18, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %20 = polygeist.submap(%0, %5, %4) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %21 = polygeist.submap(%15, %5, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %22 = polygeist.submap(%19, %5, %4) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %23 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%20, %21 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%22 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %54 = arith.subf %in, %in_2 : f64
      %55 = arith.mulf %54, %54 : f64
      %56 = arith.addf %out, %55 : f64
      linalg.yield %56 : f64
    } -> tensor<?x?xf64>
    %24 = polygeist.submapInverse(%19, %23, %5, %4) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %25 = polygeist.submap(%24, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%25 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %54 = arith.divf %out, %arg2 : f64
      %55 = math.sqrt %54 : f64
      %56 = arith.cmpf ole, %55, %cst_1 : f64
      %57 = arith.select %56, %cst, %55 : f64
      linalg.yield %57 : f64
    } -> tensor<?xf64>
    %27 = polygeist.submapInverse(%24, %26, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %28 = bufferization.to_memref %27 : memref<?xf64>
    memref.copy %28, %arg6 : memref<?xf64> to memref<?xf64>
    %29 = math.sqrt %arg2 : f64
    %30 = affine.for %arg7 = 0 to %4 iter_args(%arg8 = %0) -> (tensor<?x?xf64>) {
      %54 = affine.for %arg9 = 0 to %5 iter_args(%arg10 = %arg8) -> (tensor<?x?xf64>) {
        %extracted = tensor.extract %15[%arg9] : tensor<?xf64>
        %extracted_2 = tensor.extract %arg10[%arg7, %arg9] : tensor<?x?xf64>
        %55 = arith.subf %extracted_2, %extracted : f64
        %inserted_3 = tensor.insert %55 into %arg10[%arg7, %arg9] : tensor<?x?xf64>
        %extracted_4 = tensor.extract %27[%arg9] : tensor<?xf64>
        %56 = arith.mulf %29, %extracted_4 : f64
        %57 = arith.divf %55, %56 : f64
        %inserted_5 = tensor.insert %57 into %inserted_3[%arg7, %arg9] : tensor<?x?xf64>
        affine.yield %inserted_5 : tensor<?x?xf64>
      }
      affine.yield %54 : tensor<?x?xf64>
    }
    %31 = bufferization.to_memref %30 : memref<?x?xf64>
    memref.copy %31, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %32 = affine.apply #map4()[%5]
    %33 = polygeist.submap(%1, %32) {map = #map5} : (tensor<?x?xf64>, index) -> tensor<?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%33 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    %35 = polygeist.submapInverse(%1, %34, %32) {map = #map5} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
    %36 = affine.apply #map4()[%5]
    %37 = polygeist.submap(%35, %36, %5) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %38 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%37 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %54 = linalg.index 0 : index
      %55 = linalg.index 1 : index
      %56 = affine.apply #map6(%54)
      %57 = arith.cmpi sge, %55, %56 : index
      %58 = arith.select %57, %cst_0, %out : f64
      linalg.yield %58 : f64
    } -> tensor<?x?xf64>
    %39 = polygeist.submapInverse(%35, %38, %36, %5) {map = #map3} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %40 = affine.apply #map4()[%5]
    %41 = polygeist.submap(%30, %40, %5, %4) {map = #map7} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %42 = polygeist.submap(%30, %40, %5, %4) {map = #map8} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %43 = polygeist.submap(%39, %40, %5, %4) {map = #map9} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %44 = linalg.generic {doc = "", indexing_maps = [#map10, #map10, #map10], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%41, %42 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%43 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %54 = linalg.index 0 : index
      %55 = arith.mulf %in, %in_2 : f64
      %56 = arith.addf %out, %55 : f64
      %57 = linalg.index 1 : index
      %58 = affine.apply #map6(%54)
      %59 = arith.cmpi sge, %57, %58 : index
      %60 = arith.select %59, %56, %out : f64
      linalg.yield %60 : f64
    } -> tensor<?x?x?xf64>
    %45 = polygeist.submapInverse(%39, %44, %40, %5, %4) {map = #map9} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %46 = affine.apply #map4()[%5]
    %47 = polygeist.submap(%45, %46, %5) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %48 = polygeist.submap(%45, %46, %5) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %49 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%47 : tensor<?x?xf64>) outs(%48 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %54 = linalg.index 0 : index
      %55 = linalg.index 1 : index
      %56 = affine.apply #map6(%54)
      %57 = arith.cmpi sge, %55, %56 : index
      %58 = arith.select %57, %in, %out : f64
      linalg.yield %58 : f64
    } -> tensor<?x?xf64>
    %50 = polygeist.submapInverse(%45, %49, %46, %5) {map = #map1} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %51 = affine.apply #map4()[%5]
    %52 = affine.apply #map4()[%5]
    %inserted = tensor.insert %cst into %50[%51, %52] : tensor<?x?xf64>
    %53 = bufferization.to_memref %inserted : memref<?x?xf64>
    memref.copy %53, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

