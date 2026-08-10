#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
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
    %c2 = arith.constant 2 : index
    %c25 = arith.constant 25 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<200xf64>
    %alloca_0 = memref.alloca() : memref<200xf64>
    %alloca_1 = memref.alloca() : memref<2x4x5xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5xf64>
    %subview = memref.subview %alloca_2[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %1 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_2 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_3 = memref.subview %alloca_1[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_3 : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2, %3 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_1 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %4 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map5} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_4 = memref.subview %alloca_1[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %5 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %6 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map5} : (memref<200xf64>, index, index, index) -> memref<2x5x5xf64>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_4, %5 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%6 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %7 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map8} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_5 = memref.subview %alloca_2[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %8 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %9 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map8} : (memref<200xf64>, index, index, index) -> memref<2x5x5xf64>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_5, %8 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%9 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %alloca_6 = memref.alloca() : memref<2x4x5xf64>
    %alloca_7 = memref.alloca() : memref<2x4x5xf64>
    %subview_8 = memref.subview %alloca_7[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_8 : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %10 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %11 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%10, %11 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_7 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_9 = memref.subview %alloca_6[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_9 : memref<?x?x?xf64, strided<[20, 5, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %12 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %13 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%12, %13 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_6 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %14 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map10} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%14 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_10 = memref.subview %alloca_6[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %15 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %16 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map10} : (memref<200xf64>, index, index, index) -> memref<2x5x5xf64>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_10, %15 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%16 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %17 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map11} : (memref<200xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%17 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_11 = memref.subview %alloca_7[0, 0, 0] [%c2, %c4, %c5] [1, 1, 1] : memref<2x4x5xf64> to memref<?x?x?xf64, strided<[20, 5, 1]>>
    %18 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %19 = polygeist.submap(%alloca_0, %c2, %c5, %c5) {map = #map11} : (memref<200xf64>, index, index, index) -> memref<2x5x5xf64>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_11, %18 : memref<?x?x?xf64, strided<[20, 5, 1]>>, memref<?x?x?x?xf64>) outs(%19 : memref<2x5x5xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %20 = polygeist.submap(%arg5, %c2, %c25) {map = #map12} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %21 = polygeist.submap(%arg5, %c2, %c25) {map = #map13} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %22 = polygeist.submap(%arg5, %c2, %c25) {map = #map14} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %23 = polygeist.submap(%arg5, %c2, %c25) {map = #map15} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %24 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map12} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %25 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map13} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %26 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map14} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %27 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map15} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %subview_12 = memref.subview %arg6[0] [%c25] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %28 = polygeist.submap(%arg3, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %29 = polygeist.submap(%arg4, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %30 = polygeist.submap(%alloca, %c2, %c25) {map = #map12} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"]} ins(%20, %21, %22, %23, %24, %25, %26, %27, %subview_12, %28, %29 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64, strided<[1]>>, memref<?x?xf64>, memref<?x?xf64>) outs(%30 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_44: f64, %in_45: f64, %in_46: f64, %in_47: f64, %in_48: f64, %in_49: f64, %in_50: f64, %in_51: f64, %in_52: f64, %in_53: f64, %out: f64):
      %78 = arith.mulf %in, %in_46 : f64
      %79 = arith.mulf %in_44, %in_45 : f64
      %80 = arith.subf %78, %79 : f64
      %81 = arith.divf %in_46, %80 : f64
      %82 = arith.negf %in_44 : f64
      %83 = arith.divf %82, %80 : f64
      %84 = arith.addf %in_47, %in_50 : f64
      %85 = arith.mulf %in_51, %80 : f64
      %86 = arith.mulf %in_52, %81 : f64
      %87 = arith.mulf %86, %84 : f64
      %88 = arith.addf %in_47, %in_47 : f64
      %89 = arith.mulf %81, %88 : f64
      %90 = arith.addf %in_48, %in_49 : f64
      %91 = arith.mulf %83, %90 : f64
      %92 = arith.addf %89, %91 : f64
      %93 = arith.mulf %in_53, %92 : f64
      %94 = arith.addf %87, %93 : f64
      %95 = arith.mulf %85, %94 : f64
      linalg.yield %95 : f64
    }
    %31 = polygeist.submap(%arg5, %c2, %c25) {map = #map12} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %32 = polygeist.submap(%arg5, %c2, %c25) {map = #map13} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %33 = polygeist.submap(%arg5, %c2, %c25) {map = #map14} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %34 = polygeist.submap(%arg5, %c2, %c25) {map = #map15} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %35 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map12} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %36 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map13} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %37 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map14} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %38 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map15} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %subview_13 = memref.subview %arg6[0] [%c25] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %39 = polygeist.submap(%arg3, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %40 = polygeist.submap(%arg4, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %41 = polygeist.submap(%alloca, %c2, %c25) {map = #map14} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"]} ins(%31, %32, %33, %34, %35, %36, %37, %38, %subview_13, %39, %40 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64, strided<[1]>>, memref<?x?xf64>, memref<?x?xf64>) outs(%41 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_44: f64, %in_45: f64, %in_46: f64, %in_47: f64, %in_48: f64, %in_49: f64, %in_50: f64, %in_51: f64, %in_52: f64, %in_53: f64, %out: f64):
      %78 = arith.mulf %in, %in_46 : f64
      %79 = arith.mulf %in_44, %in_45 : f64
      %80 = arith.subf %78, %79 : f64
      %81 = arith.divf %in_46, %80 : f64
      %82 = arith.negf %in_44 : f64
      %83 = arith.divf %82, %80 : f64
      %84 = arith.addf %in_47, %in_50 : f64
      %85 = arith.mulf %in_51, %80 : f64
      %86 = arith.mulf %in_52, %83 : f64
      %87 = arith.mulf %86, %84 : f64
      %88 = arith.addf %in_49, %in_48 : f64
      %89 = arith.mulf %81, %88 : f64
      %90 = arith.addf %in_50, %in_50 : f64
      %91 = arith.mulf %83, %90 : f64
      %92 = arith.addf %89, %91 : f64
      %93 = arith.mulf %in_53, %92 : f64
      %94 = arith.addf %87, %93 : f64
      %95 = arith.mulf %85, %94 : f64
      linalg.yield %95 : f64
    }
    %42 = polygeist.submap(%arg5, %c2, %c25) {map = #map12} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %43 = polygeist.submap(%arg5, %c2, %c25) {map = #map13} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %44 = polygeist.submap(%arg5, %c2, %c25) {map = #map14} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %45 = polygeist.submap(%arg5, %c2, %c25) {map = #map15} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %46 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map12} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %47 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map13} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %48 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map14} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %49 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map15} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %subview_14 = memref.subview %arg6[0] [%c25] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %50 = polygeist.submap(%arg3, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %51 = polygeist.submap(%arg4, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %52 = polygeist.submap(%alloca, %c2, %c25) {map = #map13} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"]} ins(%42, %43, %44, %45, %46, %47, %48, %49, %subview_14, %50, %51 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64, strided<[1]>>, memref<?x?xf64>, memref<?x?xf64>) outs(%52 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_44: f64, %in_45: f64, %in_46: f64, %in_47: f64, %in_48: f64, %in_49: f64, %in_50: f64, %in_51: f64, %in_52: f64, %in_53: f64, %out: f64):
      %78 = arith.mulf %in, %in_46 : f64
      %79 = arith.mulf %in_44, %in_45 : f64
      %80 = arith.subf %78, %79 : f64
      %81 = arith.negf %in_45 : f64
      %82 = arith.divf %81, %80 : f64
      %83 = arith.divf %in, %80 : f64
      %84 = arith.addf %in_47, %in_50 : f64
      %85 = arith.mulf %in_51, %80 : f64
      %86 = arith.mulf %in_52, %82 : f64
      %87 = arith.mulf %86, %84 : f64
      %88 = arith.addf %in_47, %in_47 : f64
      %89 = arith.mulf %82, %88 : f64
      %90 = arith.addf %in_48, %in_49 : f64
      %91 = arith.mulf %83, %90 : f64
      %92 = arith.addf %89, %91 : f64
      %93 = arith.mulf %in_53, %92 : f64
      %94 = arith.addf %87, %93 : f64
      %95 = arith.mulf %85, %94 : f64
      linalg.yield %95 : f64
    }
    %53 = polygeist.submap(%arg5, %c2, %c25) {map = #map12} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %54 = polygeist.submap(%arg5, %c2, %c25) {map = #map13} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %55 = polygeist.submap(%arg5, %c2, %c25) {map = #map14} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %56 = polygeist.submap(%arg5, %c2, %c25) {map = #map15} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %57 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map12} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %58 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map13} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %59 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map14} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %60 = polygeist.submap(%alloca_0, %c2, %c25) {map = #map15} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    %subview_15 = memref.subview %arg6[0] [%c25] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %61 = polygeist.submap(%arg3, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %62 = polygeist.submap(%arg4, %c2, %c25) {map = #map16} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %63 = polygeist.submap(%alloca, %c2, %c25) {map = #map15} : (memref<200xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map17, #map17, #map17, #map17, #map17, #map17, #map17, #map17, #map18, #map17, #map17, #map17], iterator_types = ["parallel", "parallel"]} ins(%53, %54, %55, %56, %57, %58, %59, %60, %subview_15, %61, %62 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64, strided<[1]>>, memref<?x?xf64>, memref<?x?xf64>) outs(%63 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_44: f64, %in_45: f64, %in_46: f64, %in_47: f64, %in_48: f64, %in_49: f64, %in_50: f64, %in_51: f64, %in_52: f64, %in_53: f64, %out: f64):
      %78 = arith.mulf %in, %in_46 : f64
      %79 = arith.mulf %in_44, %in_45 : f64
      %80 = arith.subf %78, %79 : f64
      %81 = arith.negf %in_45 : f64
      %82 = arith.divf %81, %80 : f64
      %83 = arith.divf %in, %80 : f64
      %84 = arith.addf %in_47, %in_50 : f64
      %85 = arith.mulf %in_51, %80 : f64
      %86 = arith.mulf %in_52, %83 : f64
      %87 = arith.mulf %86, %84 : f64
      %88 = arith.addf %in_49, %in_48 : f64
      %89 = arith.mulf %82, %88 : f64
      %90 = arith.addf %in_50, %in_50 : f64
      %91 = arith.mulf %83, %90 : f64
      %92 = arith.addf %89, %91 : f64
      %93 = arith.mulf %in_53, %92 : f64
      %94 = arith.addf %87, %93 : f64
      %95 = arith.mulf %85, %94 : f64
      linalg.yield %95 : f64
    }
    %alloca_16 = memref.alloca() : memref<2x4x4xf64>
    %alloca_17 = memref.alloca() : memref<2x4x4xf64>
    %alloca_18 = memref.alloca() : memref<2x5x4xf64>
    %alloca_19 = memref.alloca() : memref<2x5x4xf64>
    %subview_20 = memref.subview %alloca_19[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_20 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %64 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map19} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %65 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%64, %65 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_19 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_21 = memref.subview %alloca_18[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_21 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %66 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map21} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %67 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%66, %67 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_18 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_22 = memref.subview %alloca_17[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_22 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_23 = memref.subview %alloca_19[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %68 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_24 = memref.subview %alloca_17[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_23, %68 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_24 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_25 = memref.subview %alloca_16[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_25 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_26 = memref.subview %alloca_18[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %69 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_27 = memref.subview %alloca_16[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_26, %69 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_27 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_28 = memref.subview %alloca_17[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %subview_29 = memref.subview %alloca_16[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %70 = polygeist.submap(%arg7, %c2, %c4, %c4) {map = #map23} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_28, %subview_29 : memref<?x?x?xf64, strided<[16, 4, 1]>>, memref<?x?x?xf64, strided<[16, 4, 1]>>) outs(%70 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.addf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %alloca_30 = memref.alloca() : memref<2x4x4xf64>
    %alloca_31 = memref.alloca() : memref<2x4x4xf64>
    %alloca_32 = memref.alloca() : memref<2x5x4xf64>
    %alloca_33 = memref.alloca() : memref<2x5x4xf64>
    %subview_34 = memref.subview %alloca_33[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_34 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %71 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map24} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %72 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%71, %72 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_33 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_35 = memref.subview %alloca_32[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_35 : memref<?x?x?xf64, strided<[20, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %73 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map25} : (memref<200xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %74 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map20} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%73, %74 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_32 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_36 = memref.subview %alloca_31[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_36 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_37 = memref.subview %alloca_33[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %75 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_38 = memref.subview %alloca_31[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_37, %75 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_38 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_39 = memref.subview %alloca_30[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview_39 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %subview_40 = memref.subview %alloca_32[0, 0, 0] [%c2, %c5, %c4] [1, 1, 1] : memref<2x5x4xf64> to memref<?x?x?xf64, strided<[20, 4, 1]>>
    %76 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map22} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %subview_41 = memref.subview %alloca_30[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    linalg.generic {indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview_40, %76 : memref<?x?x?xf64, strided<[20, 4, 1]>>, memref<?x?x?x?xf64>) outs(%subview_41 : memref<?x?x?xf64, strided<[16, 4, 1]>>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.mulf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    %subview_42 = memref.subview %alloca_31[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %subview_43 = memref.subview %alloca_30[0, 0, 0] [%c2, %c4, %c4] [1, 1, 1] : memref<2x4x4xf64> to memref<?x?x?xf64, strided<[16, 4, 1]>>
    %77 = polygeist.submap(%arg7, %c2, %c4, %c4) {map = #map26} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_42, %subview_43 : memref<?x?x?xf64, strided<[16, 4, 1]>>, memref<?x?x?xf64, strided<[16, 4, 1]>>) outs(%77 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_44: f64, %out: f64):
      %78 = arith.addf %in, %in_44 : f64
      %79 = arith.addf %out, %78 : f64
      linalg.yield %79 : f64
    }
    return
  }
}

