#map = affine_map<(d0, d1) -> (d1 + d0 * 25)>
#map1 = affine_map<(d0, d1) -> (d1 + d0 * 100)>
#map2 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 25)>
#map3 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 50)>
#map4 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 75)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_2d_scalarized(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c25 = arith.constant 25 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg5 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = bufferization.to_tensor %arg2 : memref<?xf64>
    %4 = bufferization.to_tensor %arg1 : memref<?xf64>
    %5 = bufferization.to_tensor %arg0 : memref<?xf64>
    %6 = polygeist.submap(%5, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %7 = polygeist.submap(%4, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %8 = polygeist.submap(%3, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %9 = polygeist.submap(%3, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %10 = polygeist.submap(%3, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %11 = polygeist.submap(%3, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %12 = polygeist.submap(%1, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %13 = polygeist.submap(%1, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %14 = polygeist.submap(%1, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %15 = polygeist.submap(%1, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %16 = polygeist.submap(%0, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %17 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%8, %9, %10, %11, %12, %13, %14, %15, %2, %6, %7 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%16 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %59 = arith.mulf %in, %in_2 : f64
      %60 = arith.mulf %in_0, %in_1 : f64
      %61 = arith.subf %59, %60 : f64
      %62 = arith.divf %in_2, %61 : f64
      %63 = arith.negf %in_0 : f64
      %64 = arith.divf %63, %61 : f64
      %65 = arith.addf %in_3, %in_6 : f64
      %66 = arith.mulf %in_7, %61 : f64
      %67 = arith.mulf %in_8, %62 : f64
      %68 = arith.mulf %67, %65 : f64
      %69 = arith.addf %in_3, %in_3 : f64
      %70 = arith.mulf %62, %69 : f64
      %71 = arith.addf %in_4, %in_5 : f64
      %72 = arith.mulf %64, %71 : f64
      %73 = arith.addf %70, %72 : f64
      %74 = arith.mulf %in_9, %73 : f64
      %75 = arith.addf %68, %74 : f64
      %76 = arith.mulf %66, %75 : f64
      linalg.yield %76 : f64
    } -> tensor<?x?xf64>
    %18 = polygeist.submapInverse(%0, %17, %c2, %c25) {map = #map1} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %19 = polygeist.submap(%5, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %20 = polygeist.submap(%4, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %21 = polygeist.submap(%3, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %22 = polygeist.submap(%3, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %23 = polygeist.submap(%3, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %24 = polygeist.submap(%3, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %25 = polygeist.submap(%1, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %26 = polygeist.submap(%1, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %27 = polygeist.submap(%1, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %28 = polygeist.submap(%1, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %29 = polygeist.submap(%18, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %30 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%21, %22, %23, %24, %25, %26, %27, %28, %2, %19, %20 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%29 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %59 = arith.mulf %in, %in_2 : f64
      %60 = arith.mulf %in_0, %in_1 : f64
      %61 = arith.subf %59, %60 : f64
      %62 = arith.divf %in_2, %61 : f64
      %63 = arith.negf %in_0 : f64
      %64 = arith.divf %63, %61 : f64
      %65 = arith.addf %in_3, %in_6 : f64
      %66 = arith.mulf %in_7, %61 : f64
      %67 = arith.mulf %in_8, %64 : f64
      %68 = arith.mulf %67, %65 : f64
      %69 = arith.addf %in_5, %in_4 : f64
      %70 = arith.mulf %62, %69 : f64
      %71 = arith.addf %in_6, %in_6 : f64
      %72 = arith.mulf %64, %71 : f64
      %73 = arith.addf %70, %72 : f64
      %74 = arith.mulf %in_9, %73 : f64
      %75 = arith.addf %68, %74 : f64
      %76 = arith.mulf %66, %75 : f64
      linalg.yield %76 : f64
    } -> tensor<?x?xf64>
    %31 = polygeist.submapInverse(%18, %30, %c2, %c25) {map = #map3} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %32 = polygeist.submap(%5, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %33 = polygeist.submap(%4, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %34 = polygeist.submap(%3, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %35 = polygeist.submap(%3, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %36 = polygeist.submap(%3, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %37 = polygeist.submap(%3, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %38 = polygeist.submap(%1, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %39 = polygeist.submap(%1, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %40 = polygeist.submap(%1, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %41 = polygeist.submap(%1, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %42 = polygeist.submap(%31, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %43 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%34, %35, %36, %37, %38, %39, %40, %41, %2, %32, %33 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%42 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %59 = arith.mulf %in, %in_2 : f64
      %60 = arith.mulf %in_0, %in_1 : f64
      %61 = arith.subf %59, %60 : f64
      %62 = arith.negf %in_1 : f64
      %63 = arith.divf %62, %61 : f64
      %64 = arith.divf %in, %61 : f64
      %65 = arith.addf %in_3, %in_6 : f64
      %66 = arith.mulf %in_7, %61 : f64
      %67 = arith.mulf %in_8, %63 : f64
      %68 = arith.mulf %67, %65 : f64
      %69 = arith.addf %in_3, %in_3 : f64
      %70 = arith.mulf %63, %69 : f64
      %71 = arith.addf %in_4, %in_5 : f64
      %72 = arith.mulf %64, %71 : f64
      %73 = arith.addf %70, %72 : f64
      %74 = arith.mulf %in_9, %73 : f64
      %75 = arith.addf %68, %74 : f64
      %76 = arith.mulf %66, %75 : f64
      linalg.yield %76 : f64
    } -> tensor<?x?xf64>
    %44 = polygeist.submapInverse(%31, %43, %c2, %c25) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %45 = polygeist.submap(%5, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %46 = polygeist.submap(%4, %c2, %c25) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %47 = polygeist.submap(%3, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %48 = polygeist.submap(%3, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %49 = polygeist.submap(%3, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %50 = polygeist.submap(%3, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %51 = polygeist.submap(%1, %c2, %c25) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %52 = polygeist.submap(%1, %c2, %c25) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %53 = polygeist.submap(%1, %c2, %c25) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %54 = polygeist.submap(%1, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %55 = polygeist.submap(%44, %c2, %c25) {map = #map4} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%47, %48, %49, %50, %51, %52, %53, %54, %2, %45, %46 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%55 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %59 = arith.mulf %in, %in_2 : f64
      %60 = arith.mulf %in_0, %in_1 : f64
      %61 = arith.subf %59, %60 : f64
      %62 = arith.negf %in_1 : f64
      %63 = arith.divf %62, %61 : f64
      %64 = arith.divf %in, %61 : f64
      %65 = arith.addf %in_3, %in_6 : f64
      %66 = arith.mulf %in_7, %61 : f64
      %67 = arith.mulf %in_8, %64 : f64
      %68 = arith.mulf %67, %65 : f64
      %69 = arith.addf %in_5, %in_4 : f64
      %70 = arith.mulf %63, %69 : f64
      %71 = arith.addf %in_6, %in_6 : f64
      %72 = arith.mulf %64, %71 : f64
      %73 = arith.addf %70, %72 : f64
      %74 = arith.mulf %in_9, %73 : f64
      %75 = arith.addf %68, %74 : f64
      %76 = arith.mulf %66, %75 : f64
      linalg.yield %76 : f64
    } -> tensor<?x?xf64>
    %57 = polygeist.submapInverse(%44, %56, %c2, %c25) {map = #map4} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %58 = bufferization.to_memref %57 : memref<?xf64>
    memref.copy %58, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}
