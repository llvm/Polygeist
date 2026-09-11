#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<(d0) -> (0, d0 + 1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> (d0 + 1, 0)>
#map5 = affine_map<(d0, d1) -> (d1 - 1)>
#map6 = affine_map<(d0, d1) -> (d1 + 1)>
#map7 = affine_map<(d0)[s0] -> (s0 - 1, d0 + 1)>
#map8 = affine_map<(d0, d1)[s0] -> (-(d0 + 1) + s0, d1 + 1)>
#map9 = affine_map<(d0, d1)[s0] -> (-(d0 + 1) + s0 - 1, d1 + 1)>
#map10 = affine_map<(d0, d1)[s0] -> (d1 + 1, -(d0 + 1) + s0 - 1)>
#map11 = affine_map<(d0, d1) -> (d0, d1)>
#map12 = affine_map<(d0, d1) -> (d0 - 1)>
#map13 = affine_map<(d0, d1) -> (d0 + 1)>
#map14 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map15 = affine_map<(d0, d1)[s0] -> (d1 + 1, -(d0 + 1) + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<?x20xf64>, %arg3: memref<?x20xf64>, %arg4: memref<?x20xf64>, %arg5: memref<?x20xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg5 : memref<?x20xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x20xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?x20xf64>
    %3 = bufferization.to_tensor %arg2 : memref<?x20xf64>
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
    %26:4 = affine.for %arg6 = 1 to #map()[%19] iter_args(%arg7 = %3, %arg8 = %2, %arg9 = %1, %arg10 = %0) -> (tensor<?x20xf64>, tensor<?x20xf64>, tensor<?x20xf64>, tensor<?x20xf64>) {
      %31 = affine.apply #map1()[%4]
      %32 = arith.subi %31, %c1 : index
      %33 = polygeist.submap(%arg8, %32) {map = #map2} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %34 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%33 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %35 = polygeist.submapInverse(%arg8, %34, %32) {map = #map2} : (tensor<?x20xf64>, tensor<?xf64>, index) -> tensor<?x20xf64>
      %36 = affine.apply #map1()[%4]
      %37 = arith.subi %36, %c1 : index
      %38 = polygeist.submap(%arg9, %37) {map = #map4} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %39 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%38 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %40 = polygeist.submapInverse(%arg9, %39, %37) {map = #map4} : (tensor<?x20xf64>, tensor<?xf64>, index) -> tensor<?x20xf64>
      %41 = affine.apply #map1()[%4]
      %42 = arith.subi %41, %c1 : index
      %43 = polygeist.submap(%35, %42) {map = #map2} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %44 = polygeist.submap(%arg10, %42) {map = #map4} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %45 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%43 : tensor<?xf64>) outs(%44 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %46 = polygeist.submapInverse(%arg10, %45, %42) {map = #map4} : (tensor<?x20xf64>, tensor<?xf64>, index) -> tensor<?x20xf64>
      %47:2 = affine.for %arg11 = 1 to #map1()[%4] iter_args(%arg12 = %40, %arg13 = %46) -> (tensor<?x20xf64>, tensor<?x20xf64>) {
        %107:2 = affine.for %arg14 = 1 to #map1()[%4] iter_args(%arg15 = %arg12, %arg16 = %arg13) -> (tensor<?x20xf64>, tensor<?x20xf64>) {
          %108 = affine.apply #map5(%arg11, %arg14)
          %extracted = tensor.extract %arg15[%arg11, %108] : tensor<?x20xf64>
          %109 = arith.mulf %14, %extracted : f64
          %110 = arith.addf %109, %15 : f64
          %111 = arith.divf %20, %110 : f64
          %inserted = tensor.insert %111 into %arg15[%arg11, %arg14] : tensor<?x20xf64>
          %112 = affine.apply #map5(%arg14, %arg11)
          %extracted_2 = tensor.extract %arg7[%arg14, %112] : tensor<?x20xf64>
          %113 = arith.mulf %21, %extracted_2 : f64
          %extracted_3 = tensor.extract %arg7[%arg14, %arg11] : tensor<?x20xf64>
          %114 = arith.mulf %23, %extracted_3 : f64
          %115 = arith.addf %113, %114 : f64
          %116 = affine.apply #map6(%arg14, %arg11)
          %extracted_4 = tensor.extract %arg7[%arg14, %116] : tensor<?x20xf64>
          %117 = arith.mulf %17, %extracted_4 : f64
          %118 = arith.subf %115, %117 : f64
          %119 = affine.apply #map5(%arg11, %arg14)
          %extracted_5 = tensor.extract %arg16[%arg11, %119] : tensor<?x20xf64>
          %120 = arith.mulf %14, %extracted_5 : f64
          %121 = arith.subf %118, %120 : f64
          %122 = arith.divf %121, %110 : f64
          %inserted_6 = tensor.insert %122 into %arg16[%arg11, %arg14] : tensor<?x20xf64>
          affine.yield %inserted, %inserted_6 : tensor<?x20xf64>, tensor<?x20xf64>
        }
        affine.yield %107#0, %107#1 : tensor<?x20xf64>, tensor<?x20xf64>
      }
      %48 = affine.apply #map1()[%4]
      %49 = arith.subi %48, %c1 : index
      %50 = polygeist.submap(%35, %4, %49) {map = #map7} : (tensor<?x20xf64>, index, index) -> tensor<?xf64>
      %51 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%50 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %52 = polygeist.submapInverse(%35, %51, %4, %49) {map = #map7} : (tensor<?x20xf64>, tensor<?xf64>, index, index) -> tensor<?x20xf64>
      %53 = affine.apply #map1()[%4]
      %54 = arith.subi %53, %c1 : index
      %55 = affine.apply #map1()[%4]
      %56 = arith.subi %55, %c1 : index
      %57 = affine.apply #map1()[%4]
      %58 = arith.subi %57, %c1 : index
      %59 = affine.apply #map1()[%4]
      %60 = arith.subi %59, %c1 : index
      %61 = affine.apply #map1()[%4]
      %62 = arith.subi %61, %c1 : index
      %63 = polygeist.submap(%52, %4, %58, %54) {map = #map8} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %64 = polygeist.submap(%52, %4, %62, %54) {map = #map9} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %65 = polygeist.submap(%47#0, %4, %56, %54) {map = #map10} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %66 = polygeist.submap(%47#1, %4, %60, %54) {map = #map10} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %67 = linalg.generic {doc = "", indexing_maps = [#map11, #map11, #map11, #map11], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%65, %63, %66 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%64 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64):
        %107 = arith.mulf %in, %in_2 : f64
        %108 = arith.addf %107, %in_3 : f64
        linalg.yield %108 : f64
      } -> tensor<?x?xf64>
      %68 = polygeist.submapInverse(%52, %67, %4, %62, %54) {map = #map9} : (tensor<?x20xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x20xf64>
      %69 = affine.apply #map1()[%4]
      %70 = arith.subi %69, %c1 : index
      %71 = polygeist.submap(%arg7, %70) {map = #map4} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %72 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%71 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %73 = polygeist.submapInverse(%arg7, %72, %70) {map = #map4} : (tensor<?x20xf64>, tensor<?xf64>, index) -> tensor<?x20xf64>
      %74 = affine.apply #map1()[%4]
      %75 = arith.subi %74, %c1 : index
      %76 = polygeist.submap(%47#0, %75) {map = #map4} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %77 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%76 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %78 = polygeist.submapInverse(%47#0, %77, %75) {map = #map4} : (tensor<?x20xf64>, tensor<?xf64>, index) -> tensor<?x20xf64>
      %79 = affine.apply #map1()[%4]
      %80 = arith.subi %79, %c1 : index
      %81 = polygeist.submap(%73, %80) {map = #map4} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %82 = polygeist.submap(%47#1, %80) {map = #map4} : (tensor<?x20xf64>, index) -> tensor<?xf64>
      %83 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%81 : tensor<?xf64>) outs(%82 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %84 = polygeist.submapInverse(%47#1, %83, %80) {map = #map4} : (tensor<?x20xf64>, tensor<?xf64>, index) -> tensor<?x20xf64>
      %85:2 = affine.for %arg11 = 1 to #map1()[%4] iter_args(%arg12 = %78, %arg13 = %84) -> (tensor<?x20xf64>, tensor<?x20xf64>) {
        %107:2 = affine.for %arg14 = 1 to #map1()[%4] iter_args(%arg15 = %arg12, %arg16 = %arg13) -> (tensor<?x20xf64>, tensor<?x20xf64>) {
          %108 = affine.apply #map5(%arg11, %arg14)
          %extracted = tensor.extract %arg15[%arg11, %108] : tensor<?x20xf64>
          %109 = arith.mulf %17, %extracted : f64
          %110 = arith.addf %109, %18 : f64
          %111 = arith.divf %21, %110 : f64
          %inserted = tensor.insert %111 into %arg15[%arg11, %arg14] : tensor<?x20xf64>
          %112 = affine.apply #map12(%arg11, %arg14)
          %extracted_2 = tensor.extract %68[%112, %arg14] : tensor<?x20xf64>
          %113 = arith.mulf %20, %extracted_2 : f64
          %extracted_3 = tensor.extract %68[%arg11, %arg14] : tensor<?x20xf64>
          %114 = arith.mulf %25, %extracted_3 : f64
          %115 = arith.addf %113, %114 : f64
          %116 = affine.apply #map13(%arg11, %arg14)
          %extracted_4 = tensor.extract %68[%116, %arg14] : tensor<?x20xf64>
          %117 = arith.mulf %14, %extracted_4 : f64
          %118 = arith.subf %115, %117 : f64
          %119 = affine.apply #map5(%arg11, %arg14)
          %extracted_5 = tensor.extract %arg16[%arg11, %119] : tensor<?x20xf64>
          %120 = arith.mulf %17, %extracted_5 : f64
          %121 = arith.subf %118, %120 : f64
          %122 = arith.divf %121, %110 : f64
          %inserted_6 = tensor.insert %122 into %arg16[%arg11, %arg14] : tensor<?x20xf64>
          affine.yield %inserted, %inserted_6 : tensor<?x20xf64>, tensor<?x20xf64>
        }
        affine.yield %107#0, %107#1 : tensor<?x20xf64>, tensor<?x20xf64>
      }
      %86 = affine.apply #map1()[%4]
      %87 = arith.subi %86, %c1 : index
      %88 = polygeist.submap(%73, %4, %87) {map = #map14} : (tensor<?x20xf64>, index, index) -> tensor<?xf64>
      %89 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%88 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %90 = polygeist.submapInverse(%73, %89, %4, %87) {map = #map14} : (tensor<?x20xf64>, tensor<?xf64>, index, index) -> tensor<?x20xf64>
      %91 = affine.apply #map1()[%4]
      %92 = arith.subi %91, %c1 : index
      %93 = affine.apply #map1()[%4]
      %94 = arith.subi %93, %c1 : index
      %95 = affine.apply #map1()[%4]
      %96 = arith.subi %95, %c1 : index
      %97 = affine.apply #map1()[%4]
      %98 = arith.subi %97, %c1 : index
      %99 = affine.apply #map1()[%4]
      %100 = arith.subi %99, %c1 : index
      %101 = polygeist.submap(%90, %4, %96, %92) {map = #map15} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %102 = polygeist.submap(%90, %4, %100, %92) {map = #map10} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %103 = polygeist.submap(%85#0, %4, %94, %92) {map = #map10} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %104 = polygeist.submap(%85#1, %4, %98, %92) {map = #map10} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?xf64>
      %105 = linalg.generic {doc = "", indexing_maps = [#map11, #map11, #map11, #map11], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%103, %101, %104 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%102 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64):
        %107 = arith.mulf %in, %in_2 : f64
        %108 = arith.addf %107, %in_3 : f64
        linalg.yield %108 : f64
      } -> tensor<?x?xf64>
      %106 = polygeist.submapInverse(%90, %105, %4, %100, %92) {map = #map10} : (tensor<?x20xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x20xf64>
      affine.yield %106, %68, %85#0, %85#1 : tensor<?x20xf64>, tensor<?x20xf64>, tensor<?x20xf64>, tensor<?x20xf64>
    }
    %27 = bufferization.to_memref %26#3 : memref<?x20xf64>
    memref.copy %27, %arg5 : memref<?x20xf64> to memref<?x20xf64>
    %28 = bufferization.to_memref %26#2 : memref<?x20xf64>
    memref.copy %28, %arg4 : memref<?x20xf64> to memref<?x20xf64>
    %29 = bufferization.to_memref %26#1 : memref<?x20xf64>
    memref.copy %29, %arg3 : memref<?x20xf64> to memref<?x20xf64>
    %30 = bufferization.to_memref %26#0 : memref<?x20xf64>
    memref.copy %30, %arg2 : memref<?x20xf64> to memref<?x20xf64>
    return
  }
}

