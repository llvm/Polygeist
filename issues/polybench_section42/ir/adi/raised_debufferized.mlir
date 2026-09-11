#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0)[s0] -> (s0 - 1, d0 + 1)>
#map6 = affine_map<(d0, d1)[s0] -> (d0 + 1, -(d1 + 1) + s0 - 1)>
#map7 = affine_map<(d0, d1)[s0] -> (-(d1 + 1) + s0, d0 + 1)>
#map8 = affine_map<(d0, d1)[s0] -> (-(d1 + 1) + s0 - 1, d0 + 1)>
#map9 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map10 = affine_map<(d0, d1)[s0] -> (d0 + 1, -(d1 + 1) + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg5 : memref<?x?xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.sitofp %arg1 : i32 to f64
    %6 = arith.divf %cst_1, %5 : f64
    %7 = arith.sitofp %arg0 : i32 to f64
    %8 = arith.divf %cst_1, %7 : f64
    %9 = arith.mulf %8, %cst_0 : f64
    %10 = arith.mulf %6, %6 : f64
    %11 = arith.divf %9, %10 : f64
    %12 = arith.divf %8, %10 : f64
    %13 = arith.negf %11 : f64
    %14 = arith.divf %13, %cst_0 : f64
    %15 = arith.addf %11, %cst_1 : f64
    %16 = arith.negf %12 : f64
    %17 = arith.divf %16, %cst_0 : f64
    %18 = arith.addf %12, %cst_1 : f64
    %19 = arith.index_cast %arg0 : i32 to index
    %20 = arith.negf %14 : f64
    %21 = arith.negf %17 : f64
    %22 = arith.mulf %17, %cst_0 : f64
    %23 = arith.addf %22, %cst_1 : f64
    %24 = arith.mulf %14, %cst_0 : f64
    %25 = arith.addf %24, %cst_1 : f64
    %26 = affine.apply #map()[%4]
    %27 = affine.apply #map()[%4]
    %28 = arith.subi %27, %c1 : index
    %29 = affine.apply #map()[%4]
    %30 = arith.subi %29, %c1 : index
    %31 = affine.apply #map()[%4]
    %32 = arith.subi %31, %c1 : index
    %33 = affine.apply #map()[%4]
    %34 = arith.subi %33, %c1 : index
    %35 = affine.apply #map()[%4]
    %36:4 = affine.for %arg6 = 1 to #map1()[%19] iter_args(%arg7 = %1, %arg8 = %2, %arg9 = %3, %arg10 = %0) -> (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) {
      %extracted_slice = tensor.extract_slice %arg7[0, 1] [1, %32] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %41 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %inserted_slice = tensor.insert_slice %41 into %arg7[0, 1] [1, %32] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %extracted_slice_2 = tensor.extract_slice %arg8[1, 0] [%34, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %42 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice_2 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %inserted_slice_3 = tensor.insert_slice %42 into %arg8[1, 0] [%34, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %43 = arith.subi %35, %c1 : index
      %extracted_slice_4 = tensor.extract_slice %inserted_slice[0, 1] [1, %43] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_5 = tensor.extract_slice %arg9[1, 0] [%43, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %44 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_4 : tensor<?xf64>) outs(%extracted_slice_5 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %inserted_slice_6 = tensor.insert_slice %44 into %arg9[1, 0] [%43, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %45 = affine.apply #map()[%4]
      %46 = arith.subi %45, %c1 : index
      %extracted_slice_7 = tensor.extract_slice %inserted_slice_3[1, 0] [%46, %28] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_8 = tensor.extract_slice %arg10[1, 0] [%28, %46] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_9 = tensor.extract_slice %arg10[1, 1] [%28, %46] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_10 = tensor.extract_slice %arg10[1, 2] [%28, %46] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_11 = tensor.extract_slice %inserted_slice_6[1, 0] [%46, %28] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_12 = tensor.extract_slice %inserted_slice_3[1, 1] [%46, %28] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_13 = tensor.extract_slice %inserted_slice_6[1, 1] [%46, %28] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %47:2 = linalg.generic {doc = "", indexing_maps = [#map3, #map4, #map4, #map4, #map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_7, %extracted_slice_8, %extracted_slice_9, %extracted_slice_10, %extracted_slice_11 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_12, %extracted_slice_13 : tensor<?x?xf64>, tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_32: f64, %in_33: f64, %in_34: f64, %in_35: f64, %out: f64, %out_36: f64):
        %89 = arith.mulf %14, %in : f64
        %90 = arith.addf %89, %15 : f64
        %91 = arith.divf %20, %90 : f64
        %92 = arith.mulf %21, %in_32 : f64
        %93 = arith.mulf %23, %in_33 : f64
        %94 = arith.addf %92, %93 : f64
        %95 = arith.mulf %17, %in_34 : f64
        %96 = arith.subf %94, %95 : f64
        %97 = arith.mulf %14, %in_35 : f64
        %98 = arith.subf %96, %97 : f64
        %99 = arith.divf %98, %90 : f64
        linalg.yield %91, %99 : f64, f64
      } -> (tensor<?x?xf64>, tensor<?x?xf64>)
      %inserted_slice_14 = tensor.insert_slice %47#0 into %inserted_slice_3[1, 1] [%46, %28] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %inserted_slice_15 = tensor.insert_slice %47#1 into %inserted_slice_6[1, 1] [%46, %28] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %48 = affine.apply #map()[%4]
      %49 = arith.subi %48, %c1 : index
      %50 = polygeist.submap(%inserted_slice, %4, %49) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %51 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%50 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %52 = polygeist.submapInverse(%inserted_slice, %51, %4, %49) {map = #map5} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %53 = affine.apply #map()[%4]
      %54 = arith.subi %53, %c1 : index
      %55 = polygeist.submap(%inserted_slice_14, %4, %54, %30) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %56 = polygeist.submap(%52, %4, %54, %30) {map = #map7} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %57 = polygeist.submap(%inserted_slice_15, %4, %54, %30) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %58 = polygeist.submap(%52, %4, %54, %30) {map = #map8} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %59 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%55, %56, %57 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%58 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_32: f64, %in_33: f64, %out: f64):
        %89 = arith.mulf %in, %in_32 : f64
        %90 = arith.addf %89, %in_33 : f64
        linalg.yield %90 : f64
      } -> tensor<?x?xf64>
      %60 = polygeist.submapInverse(%52, %59, %4, %54, %30) {map = #map8} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %61 = arith.subi %26, %c1 : index
      %62 = affine.apply #map()[%4]
      %63 = arith.subi %62, %c1 : index
      %64 = affine.apply #map()[%4]
      %65 = arith.subi %64, %c1 : index
      %extracted_slice_16 = tensor.extract_slice %arg10[1, 0] [%65, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %66 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice_16 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %inserted_slice_17 = tensor.insert_slice %66 into %arg10[1, 0] [%65, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %67 = affine.apply #map()[%4]
      %68 = arith.subi %67, %c1 : index
      %extracted_slice_18 = tensor.extract_slice %inserted_slice_14[1, 0] [%68, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %69 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice_18 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %inserted_slice_19 = tensor.insert_slice %69 into %inserted_slice_14[1, 0] [%68, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %70 = affine.apply #map()[%4]
      %71 = arith.subi %70, %c1 : index
      %extracted_slice_20 = tensor.extract_slice %inserted_slice_17[1, 0] [%71, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_21 = tensor.extract_slice %inserted_slice_15[1, 0] [%71, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %72 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_20 : tensor<?xf64>) outs(%extracted_slice_21 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %inserted_slice_22 = tensor.insert_slice %72 into %inserted_slice_15[1, 0] [%71, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %73 = affine.apply #map()[%4]
      %74 = arith.subi %73, %c1 : index
      %extracted_slice_23 = tensor.extract_slice %inserted_slice_19[1, 0] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_24 = tensor.extract_slice %60[0, 1] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_25 = tensor.extract_slice %60[1, 1] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_26 = tensor.extract_slice %60[2, 1] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_27 = tensor.extract_slice %inserted_slice_22[1, 0] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_28 = tensor.extract_slice %inserted_slice_19[1, 1] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %extracted_slice_29 = tensor.extract_slice %inserted_slice_22[1, 1] [%74, %61] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %75:2 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3, #map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_23, %extracted_slice_24, %extracted_slice_25, %extracted_slice_26, %extracted_slice_27 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_28, %extracted_slice_29 : tensor<?x?xf64>, tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_32: f64, %in_33: f64, %in_34: f64, %in_35: f64, %out: f64, %out_36: f64):
        %89 = arith.mulf %17, %in : f64
        %90 = arith.addf %89, %18 : f64
        %91 = arith.divf %21, %90 : f64
        %92 = arith.mulf %20, %in_32 : f64
        %93 = arith.mulf %25, %in_33 : f64
        %94 = arith.addf %92, %93 : f64
        %95 = arith.mulf %14, %in_34 : f64
        %96 = arith.subf %94, %95 : f64
        %97 = arith.mulf %17, %in_35 : f64
        %98 = arith.subf %96, %97 : f64
        %99 = arith.divf %98, %90 : f64
        linalg.yield %91, %99 : f64, f64
      } -> (tensor<?x?xf64>, tensor<?x?xf64>)
      %inserted_slice_30 = tensor.insert_slice %75#0 into %inserted_slice_19[1, 1] [%74, %61] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %inserted_slice_31 = tensor.insert_slice %75#1 into %inserted_slice_22[1, 1] [%74, %61] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      %76 = affine.apply #map()[%4]
      %77 = arith.subi %76, %c1 : index
      %78 = polygeist.submap(%inserted_slice_17, %4, %77) {map = #map9} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %79 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%78 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %80 = polygeist.submapInverse(%inserted_slice_17, %79, %4, %77) {map = #map9} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %81 = affine.apply #map()[%4]
      %82 = arith.subi %81, %c1 : index
      %83 = polygeist.submap(%inserted_slice_30, %4, %82, %63) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %84 = polygeist.submap(%80, %4, %82, %63) {map = #map10} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %85 = polygeist.submap(%inserted_slice_31, %4, %82, %63) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %86 = polygeist.submap(%80, %4, %82, %63) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %87 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%83, %84, %85 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%86 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_32: f64, %in_33: f64, %out: f64):
        %89 = arith.mulf %in, %in_32 : f64
        %90 = arith.addf %89, %in_33 : f64
        linalg.yield %90 : f64
      } -> tensor<?x?xf64>
      %88 = polygeist.submapInverse(%80, %87, %4, %82, %63) {map = #map6} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      affine.yield %60, %inserted_slice_30, %inserted_slice_31, %88 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>
    }
    %37 = bufferization.to_memref %36#1 : memref<?x?xf64>
    memref.copy %37, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    %38 = bufferization.to_memref %36#2 : memref<?x?xf64>
    memref.copy %38, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %39 = bufferization.to_memref %36#3 : memref<?x?xf64>
    memref.copy %39, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    %40 = bufferization.to_memref %36#0 : memref<?x?xf64>
    memref.copy %40, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

