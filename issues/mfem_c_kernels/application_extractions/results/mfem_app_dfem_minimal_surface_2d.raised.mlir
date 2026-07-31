#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 * 5 + d1 + d0 * 50 + 25)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d3 * 4 + d1)>
#map15 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_dfem_minimal_surface_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<100xf64>
    %alloca_1 = memref.alloca() : memref<100xf64>
    %alloca_2 = memref.alloca() : memref<2x4x5xf64>
    %alloca_3 = memref.alloca() : memref<2x4x5xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_3 : memref<2x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %0 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %1 = polygeist.submap(%arg0, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_3 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_2 : memref<2x4x5xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = polygeist.submap(%arg2, %c2, %c4, %c5, %c4) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %3 = polygeist.submap(%arg1, %c2, %c4, %c5, %c4) {map = #map2} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2, %3 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_2 : memref<2x4x5xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    %4 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map5} : (memref<100xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %5 = polygeist.submap(%arg0, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %6 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4) {map = #map7} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_2, %5 : memref<2x4x5xf64>, memref<?x?x?x?xf64>) outs(%6 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    %7 = polygeist.submap(%alloca_1, %c2, %c5, %c5) {map = #map9} : (memref<100xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : memref<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %8 = polygeist.submap(%arg1, %c2, %c5, %c5, %c4) {map = #map6} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %9 = polygeist.submap(%alloca_1, %c2, %c5, %c5, %c4) {map = #map10} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_3, %8 : memref<2x4x5xf64>, memref<?x?x?x?xf64>) outs(%9 : memref<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    affine.for %arg6 = 0 to 5 {
      affine.for %arg7 = 0 to 5 {
        %17 = affine.load %alloca_1[%arg7 + %arg6 * 5] : memref<100xf64>
        %18 = affine.load %alloca_1[%arg7 + %arg6 * 5 + 25] : memref<100xf64>
        %19 = affine.load %arg3[%arg7 * 4 + %arg6 * 20] : memref<?xf64>
        %20 = affine.load %arg3[%arg7 * 4 + %arg6 * 20 + 1] : memref<?xf64>
        %21 = affine.load %arg3[%arg7 * 4 + %arg6 * 20 + 2] : memref<?xf64>
        %22 = affine.load %arg3[%arg7 * 4 + %arg6 * 20 + 3] : memref<?xf64>
        %23 = arith.mulf %19, %22 : f64
        %24 = arith.mulf %20, %21 : f64
        %25 = arith.subf %23, %24 : f64
        %26 = arith.divf %22, %25 : f64
        %27 = arith.negf %20 : f64
        %28 = arith.divf %27, %25 : f64
        %29 = arith.negf %21 : f64
        %30 = arith.divf %29, %25 : f64
        %31 = arith.divf %19, %25 : f64
        %32 = arith.mulf %17, %26 : f64
        %33 = arith.mulf %18, %30 : f64
        %34 = arith.addf %32, %33 : f64
        %35 = arith.mulf %17, %28 : f64
        %36 = arith.mulf %18, %31 : f64
        %37 = arith.addf %35, %36 : f64
        %38 = arith.mulf %34, %34 : f64
        %39 = arith.addf %38, %cst_0 : f64
        %40 = arith.mulf %37, %37 : f64
        %41 = arith.addf %39, %40 : f64
        %42 = math.sqrt %41 : f64
        %43 = arith.divf %cst_0, %42 : f64
        %44 = arith.mulf %43, %25 : f64
        %45 = affine.load %arg4[%arg7 + %arg6 * 5] : memref<?xf64>
        %46 = arith.mulf %44, %45 : f64
        %47 = arith.mulf %34, %26 : f64
        %48 = arith.mulf %37, %28 : f64
        %49 = arith.addf %47, %48 : f64
        %50 = arith.mulf %46, %49 : f64
        affine.store %50, %alloca[%arg7 + %arg6 * 5] : memref<100xf64>
        %51 = arith.mulf %34, %30 : f64
        %52 = arith.mulf %37, %31 : f64
        %53 = arith.addf %51, %52 : f64
        %54 = arith.mulf %46, %53 : f64
        affine.store %54, %alloca[%arg7 + %arg6 * 5 + 25] : memref<100xf64>
        affine.store %cst, %alloca[%arg7 + %arg6 * 5 + 50] : memref<100xf64>
        affine.store %cst, %alloca[%arg7 + %arg6 * 5 + 75] : memref<100xf64>
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    %alloca_4 = memref.alloca() : memref<2x4x4xf64>
    %alloca_5 = memref.alloca() : memref<2x4x4xf64>
    %alloca_6 = memref.alloca() : memref<2x5x4xf64>
    %alloca_7 = memref.alloca() : memref<2x5x4xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_7 : memref<2x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %10 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map11} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %11 = polygeist.submap(%arg1, %c2, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%10, %11 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_7 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_6 : memref<2x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %12 = polygeist.submap(%alloca, %c2, %c5, %c4, %c5) {map = #map13} : (memref<100xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    %13 = polygeist.submap(%arg0, %c2, %c5, %c4, %c5) {map = #map12} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%12, %13 : memref<?x?x?x?xf64>, memref<?x?x?x?xf64>) outs(%alloca_6 : memref<2x5x4xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_5 : memref<2x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %14 = polygeist.submap(%arg0, %c2, %c4, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_7, %14 : memref<2x5x4xf64>, memref<?x?x?x?xf64>) outs(%alloca_5 : memref<2x4x4xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_4 : memref<2x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %15 = polygeist.submap(%arg1, %c2, %c4, %c4, %c5) {map = #map14} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?x?xf64>
    linalg.generic {indexing_maps = [#map8, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloca_6, %15 : memref<2x5x4xf64>, memref<?x?x?x?xf64>) outs(%alloca_4 : memref<2x4x4xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.mulf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    %16 = polygeist.submap(%arg5, %c2, %c4, %c4) {map = #map15} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloca_5, %alloca_4 : memref<2x4x4xf64>, memref<2x4x4xf64>) outs(%16 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %17 = arith.addf %in, %in_8 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    return
  }
}
