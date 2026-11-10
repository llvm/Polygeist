#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<(d0) -> (0, d0 + 1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> (d0 + 1, 0)>
#map5 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map6 = affine_map<(d0, d1) -> (d1 + 1, d0)>
#map7 = affine_map<(d0, d1) -> (d1 + 1, d0 + 1)>
#map8 = affine_map<(d0, d1) -> (d1 + 1, d0 + 2)>
#map9 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map10 = affine_map<(d0, d1) -> (d0, d1)>
#map11 = affine_map<(d0)[s0] -> (s0 - 1, d0 + 1)>
#map12 = affine_map<(d0, d1)[s0] -> (d0 + 1, -(d1 + 1) + s0 - 1)>
#map13 = affine_map<(d0, d1)[s0] -> (-(d1 + 1) + s0, d0 + 1)>
#map14 = affine_map<(d0, d1)[s0] -> (-(d1 + 1) + s0 - 1, d0 + 1)>
#map15 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map16 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
#map17 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map18 = affine_map<(d0, d1)[s0] -> (d0 + 1, -(d1 + 1) + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      %41 = polygeist.submap(%arg7, %32) {map = #map2} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %42 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%41 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %43 = polygeist.submapInverse(%arg7, %42, %32) {map = #map2} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %44 = polygeist.submap(%arg8, %34) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %45 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%44 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %46 = polygeist.submapInverse(%arg8, %45, %34) {map = #map4} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %47 = arith.subi %35, %c1 : index
      %48 = polygeist.submap(%43, %47) {map = #map2} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %49 = polygeist.submap(%arg9, %47) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %50 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%48 : tensor<?xf64>) outs(%49 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %51 = polygeist.submapInverse(%arg9, %50, %47) {map = #map4} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %52 = affine.apply #map()[%4]
      %53 = arith.subi %52, %c1 : index
      %54 = polygeist.submap(%46, %53, %28) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %55 = polygeist.submap(%arg10, %53, %28) {map = #map6} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %56 = polygeist.submap(%arg10, %53, %28) {map = #map7} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %57 = polygeist.submap(%arg10, %53, %28) {map = #map8} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %58 = polygeist.submap(%51, %53, %28) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %59 = polygeist.submap(%46, %53, %28) {map = #map9} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %60 = polygeist.submap(%51, %53, %28) {map = #map9} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %61:2 = linalg.generic {doc = "", indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%54, %55, %56, %57, %58 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%59, %60 : tensor<?x?xf64>, tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64, %out_6: f64):
        %121 = arith.mulf %14, %in : f64
        %122 = arith.addf %121, %15 : f64
        %123 = arith.divf %20, %122 : f64
        %124 = arith.mulf %21, %in_2 : f64
        %125 = arith.mulf %23, %in_3 : f64
        %126 = arith.addf %124, %125 : f64
        %127 = arith.mulf %17, %in_4 : f64
        %128 = arith.subf %126, %127 : f64
        %129 = arith.mulf %14, %in_5 : f64
        %130 = arith.subf %128, %129 : f64
        %131 = arith.divf %130, %122 : f64
        linalg.yield %123, %131 : f64, f64
      } -> (tensor<?x?xf64>, tensor<?x?xf64>)
      %62 = polygeist.submapInverse(%46, %61#0, %53, %28) {map = #map9} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %63 = polygeist.submapInverse(%51, %61#1, %53, %28) {map = #map9} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %64 = affine.apply #map()[%4]
      %65 = arith.subi %64, %c1 : index
      %66 = polygeist.submap(%43, %4, %65) {map = #map11} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %67 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%66 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %68 = polygeist.submapInverse(%43, %67, %4, %65) {map = #map11} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %69 = affine.apply #map()[%4]
      %70 = arith.subi %69, %c1 : index
      %71 = polygeist.submap(%62, %4, %70, %30) {map = #map12} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %72 = polygeist.submap(%68, %4, %70, %30) {map = #map13} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %73 = polygeist.submap(%63, %4, %70, %30) {map = #map12} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %74 = polygeist.submap(%68, %4, %70, %30) {map = #map14} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %75 = linalg.generic {doc = "", indexing_maps = [#map10, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%71, %72, %73 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%74 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64):
        %121 = arith.mulf %in, %in_2 : f64
        %122 = arith.addf %121, %in_3 : f64
        linalg.yield %122 : f64
      } -> tensor<?x?xf64>
      %76 = polygeist.submapInverse(%68, %75, %4, %70, %30) {map = #map14} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %77 = arith.subi %26, %c1 : index
      %78 = affine.apply #map()[%4]
      %79 = arith.subi %78, %c1 : index
      %80 = affine.apply #map()[%4]
      %81 = arith.subi %80, %c1 : index
      %82 = polygeist.submap(%arg10, %81) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %83 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%82 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %84 = polygeist.submapInverse(%arg10, %83, %81) {map = #map4} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %85 = affine.apply #map()[%4]
      %86 = arith.subi %85, %c1 : index
      %87 = polygeist.submap(%62, %86) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %88 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%87 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %89 = polygeist.submapInverse(%62, %88, %86) {map = #map4} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %90 = affine.apply #map()[%4]
      %91 = arith.subi %90, %c1 : index
      %92 = polygeist.submap(%84, %91) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %93 = polygeist.submap(%63, %91) {map = #map4} : (tensor<?x?xf64>, index) -> tensor<?xf64>
      %94 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%92 : tensor<?xf64>) outs(%93 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      } -> tensor<?xf64>
      %95 = polygeist.submapInverse(%63, %94, %91) {map = #map4} : (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
      %96 = affine.apply #map()[%4]
      %97 = arith.subi %96, %c1 : index
      %98 = polygeist.submap(%89, %97, %77) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %99 = polygeist.submap(%76, %97, %77) {map = #map15} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %100 = polygeist.submap(%76, %97, %77) {map = #map9} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %101 = polygeist.submap(%76, %97, %77) {map = #map16} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %102 = polygeist.submap(%95, %97, %77) {map = #map5} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %103 = polygeist.submap(%89, %97, %77) {map = #map9} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %104 = polygeist.submap(%95, %97, %77) {map = #map9} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %105:2 = linalg.generic {doc = "", indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%98, %99, %100, %101, %102 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%103, %104 : tensor<?x?xf64>, tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64, %out_6: f64):
        %121 = arith.mulf %17, %in : f64
        %122 = arith.addf %121, %18 : f64
        %123 = arith.divf %21, %122 : f64
        %124 = arith.mulf %20, %in_2 : f64
        %125 = arith.mulf %25, %in_3 : f64
        %126 = arith.addf %124, %125 : f64
        %127 = arith.mulf %14, %in_4 : f64
        %128 = arith.subf %126, %127 : f64
        %129 = arith.mulf %17, %in_5 : f64
        %130 = arith.subf %128, %129 : f64
        %131 = arith.divf %130, %122 : f64
        linalg.yield %123, %131 : f64, f64
      } -> (tensor<?x?xf64>, tensor<?x?xf64>)
      %106 = polygeist.submapInverse(%89, %105#0, %97, %77) {map = #map9} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %107 = polygeist.submapInverse(%95, %105#1, %97, %77) {map = #map9} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %108 = affine.apply #map()[%4]
      %109 = arith.subi %108, %c1 : index
      %110 = polygeist.submap(%84, %4, %109) {map = #map17} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %111 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%110 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      } -> tensor<?xf64>
      %112 = polygeist.submapInverse(%84, %111, %4, %109) {map = #map17} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %113 = affine.apply #map()[%4]
      %114 = arith.subi %113, %c1 : index
      %115 = polygeist.submap(%106, %4, %114, %79) {map = #map12} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %116 = polygeist.submap(%112, %4, %114, %79) {map = #map18} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %117 = polygeist.submap(%107, %4, %114, %79) {map = #map12} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %118 = polygeist.submap(%112, %4, %114, %79) {map = #map12} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %119 = linalg.generic {doc = "", indexing_maps = [#map10, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%115, %116, %117 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%118 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64):
        %121 = arith.mulf %in, %in_2 : f64
        %122 = arith.addf %121, %in_3 : f64
        linalg.yield %122 : f64
      } -> tensor<?x?xf64>
      %120 = polygeist.submapInverse(%112, %119, %4, %114, %79) {map = #map12} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      affine.yield %76, %106, %107, %120 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>
    }
    %37 = bufferization.to_memref %36#1 : memref<?x?xf64>
    memref.copy %37, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    %38 = bufferization.to_memref %36#0 : memref<?x?xf64>
    memref.copy %38, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %39 = bufferization.to_memref %36#2 : memref<?x?xf64>
    memref.copy %39, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %40 = bufferization.to_memref %36#3 : memref<?x?xf64>
    memref.copy %40, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

