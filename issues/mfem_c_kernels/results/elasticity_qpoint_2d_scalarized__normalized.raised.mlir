#map = affine_map<(d0, d1) -> (d1 + d0 * 100)>
#map1 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 25)>
#map2 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 50)>
#map3 = affine_map<(d0, d1) -> (d1 + d0 * 100 + 75)>
#map4 = affine_map<(d0, d1) -> (d1 + d0 * 25)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_2d_scalarized(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c25 = arith.constant 25 : index
    %0 = polygeist.submap(%arg2, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %1 = polygeist.submap(%arg2, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %2 = polygeist.submap(%arg2, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %3 = polygeist.submap(%arg2, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %4 = polygeist.submap(%arg4, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %5 = polygeist.submap(%arg4, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %6 = polygeist.submap(%arg4, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %7 = polygeist.submap(%arg4, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %8 = polygeist.submap(%arg0, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %9 = polygeist.submap(%arg1, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %10 = polygeist.submap(%arg5, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %2, %3, %4, %5, %6, %7, %arg3, %8, %9 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%10 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %44 = arith.mulf %in, %in_2 : f64
      %45 = arith.mulf %in_0, %in_1 : f64
      %46 = arith.subf %44, %45 : f64
      %47 = arith.divf %in_2, %46 : f64
      %48 = arith.negf %in_0 : f64
      %49 = arith.divf %48, %46 : f64
      %50 = arith.addf %in_3, %in_6 : f64
      %51 = arith.mulf %in_7, %46 : f64
      %52 = arith.mulf %in_8, %47 : f64
      %53 = arith.mulf %52, %50 : f64
      %54 = arith.addf %in_3, %in_3 : f64
      %55 = arith.mulf %47, %54 : f64
      %56 = arith.addf %in_4, %in_5 : f64
      %57 = arith.mulf %49, %56 : f64
      %58 = arith.addf %55, %57 : f64
      %59 = arith.mulf %in_9, %58 : f64
      %60 = arith.addf %53, %59 : f64
      %61 = arith.mulf %51, %60 : f64
      linalg.yield %61 : f64
    }
    %11 = polygeist.submap(%arg2, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %12 = polygeist.submap(%arg2, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %13 = polygeist.submap(%arg2, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %14 = polygeist.submap(%arg2, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %15 = polygeist.submap(%arg4, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %16 = polygeist.submap(%arg4, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %17 = polygeist.submap(%arg4, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %18 = polygeist.submap(%arg4, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %19 = polygeist.submap(%arg0, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %20 = polygeist.submap(%arg1, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %21 = polygeist.submap(%arg5, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%11, %12, %13, %14, %15, %16, %17, %18, %arg3, %19, %20 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%21 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %44 = arith.mulf %in, %in_2 : f64
      %45 = arith.mulf %in_0, %in_1 : f64
      %46 = arith.subf %44, %45 : f64
      %47 = arith.divf %in_2, %46 : f64
      %48 = arith.negf %in_0 : f64
      %49 = arith.divf %48, %46 : f64
      %50 = arith.addf %in_3, %in_6 : f64
      %51 = arith.mulf %in_7, %46 : f64
      %52 = arith.mulf %in_8, %49 : f64
      %53 = arith.mulf %52, %50 : f64
      %54 = arith.addf %in_5, %in_4 : f64
      %55 = arith.mulf %47, %54 : f64
      %56 = arith.addf %in_6, %in_6 : f64
      %57 = arith.mulf %49, %56 : f64
      %58 = arith.addf %55, %57 : f64
      %59 = arith.mulf %in_9, %58 : f64
      %60 = arith.addf %53, %59 : f64
      %61 = arith.mulf %51, %60 : f64
      linalg.yield %61 : f64
    }
    %22 = polygeist.submap(%arg2, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %23 = polygeist.submap(%arg2, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %24 = polygeist.submap(%arg2, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %25 = polygeist.submap(%arg2, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %26 = polygeist.submap(%arg4, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %27 = polygeist.submap(%arg4, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %28 = polygeist.submap(%arg4, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %29 = polygeist.submap(%arg4, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %30 = polygeist.submap(%arg0, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %31 = polygeist.submap(%arg1, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %32 = polygeist.submap(%arg5, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%22, %23, %24, %25, %26, %27, %28, %29, %arg3, %30, %31 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%32 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %44 = arith.mulf %in, %in_2 : f64
      %45 = arith.mulf %in_0, %in_1 : f64
      %46 = arith.subf %44, %45 : f64
      %47 = arith.negf %in_1 : f64
      %48 = arith.divf %47, %46 : f64
      %49 = arith.divf %in, %46 : f64
      %50 = arith.addf %in_3, %in_6 : f64
      %51 = arith.mulf %in_7, %46 : f64
      %52 = arith.mulf %in_8, %48 : f64
      %53 = arith.mulf %52, %50 : f64
      %54 = arith.addf %in_3, %in_3 : f64
      %55 = arith.mulf %48, %54 : f64
      %56 = arith.addf %in_4, %in_5 : f64
      %57 = arith.mulf %49, %56 : f64
      %58 = arith.addf %55, %57 : f64
      %59 = arith.mulf %in_9, %58 : f64
      %60 = arith.addf %53, %59 : f64
      %61 = arith.mulf %51, %60 : f64
      linalg.yield %61 : f64
    }
    %33 = polygeist.submap(%arg2, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %34 = polygeist.submap(%arg2, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %35 = polygeist.submap(%arg2, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %36 = polygeist.submap(%arg2, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %37 = polygeist.submap(%arg4, %c2, %c25) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %38 = polygeist.submap(%arg4, %c2, %c25) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %39 = polygeist.submap(%arg4, %c2, %c25) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %40 = polygeist.submap(%arg4, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %41 = polygeist.submap(%arg0, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %42 = polygeist.submap(%arg1, %c2, %c25) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %43 = polygeist.submap(%arg5, %c2, %c25) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5, #map5, #map5, #map5, #map5, #map6, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%33, %34, %35, %36, %37, %38, %39, %40, %arg3, %41, %42 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%43 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %out: f64):
      %44 = arith.mulf %in, %in_2 : f64
      %45 = arith.mulf %in_0, %in_1 : f64
      %46 = arith.subf %44, %45 : f64
      %47 = arith.negf %in_1 : f64
      %48 = arith.divf %47, %46 : f64
      %49 = arith.divf %in, %46 : f64
      %50 = arith.addf %in_3, %in_6 : f64
      %51 = arith.mulf %in_7, %46 : f64
      %52 = arith.mulf %in_8, %49 : f64
      %53 = arith.mulf %52, %50 : f64
      %54 = arith.addf %in_5, %in_4 : f64
      %55 = arith.mulf %48, %54 : f64
      %56 = arith.addf %in_6, %in_6 : f64
      %57 = arith.mulf %49, %56 : f64
      %58 = arith.addf %55, %57 : f64
      %59 = arith.mulf %in_9, %58 : f64
      %60 = arith.addf %53, %59 : f64
      %61 = arith.mulf %51, %60 : f64
      linalg.yield %61 : f64
    }
    return
  }
}
