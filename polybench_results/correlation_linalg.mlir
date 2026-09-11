#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<()[s0] -> (s0 - 1)>
#map4 = affine_map<(d0) -> (d0, d0)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0) -> (d0 + 1)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x28xf64>, %arg4: memref<?x28xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = polygeist.submap(%arg5, %1) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : memref<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    }
    %3 = polygeist.submap(%arg3, %0, %1) {map = #map1} : (memref<?x28xf64>, index, index) -> memref<?x?xf64>
    %4 = polygeist.submap(%arg5, %0, %1) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : memref<?x?xf64>) outs(%4 : memref<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %23 = arith.addf %out, %in : f64
      linalg.yield %23 : f64
    }
    %5 = polygeist.submap(%arg5, %1) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%5 : memref<?xf64>) {
    ^bb0(%out: f64):
      %23 = arith.divf %out, %arg2 : f64
      linalg.yield %23 : f64
    }
    %6 = polygeist.submap(%arg6, %1) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : memref<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    }
    %7 = polygeist.submap(%arg3, %0, %1) {map = #map1} : (memref<?x28xf64>, index, index) -> memref<?x?xf64>
    %8 = polygeist.submap(%arg5, %0, %1) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %9 = polygeist.submap(%arg6, %0, %1) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%7, %8 : memref<?x?xf64>, memref<?x?xf64>) outs(%9 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %23 = arith.subf %in, %in_2 : f64
      %24 = arith.mulf %23, %23 : f64
      %25 = arith.addf %out, %24 : f64
      linalg.yield %25 : f64
    }
    %10 = polygeist.submap(%arg6, %1) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%10 : memref<?xf64>) {
    ^bb0(%out: f64):
      %23 = arith.divf %out, %arg2 : f64
      %24 = math.sqrt %23 : f64
      %25 = arith.cmpf ole, %24, %cst : f64
      %26 = arith.select %25, %cst_1, %24 : f64
      linalg.yield %26 : f64
    }
    %11 = math.sqrt %arg2 : f64
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to %1 {
        %23 = affine.load %arg5[%arg8] : memref<?xf64>
        %24 = affine.load %arg3[%arg7, %arg8] : memref<?x28xf64>
        %25 = arith.subf %24, %23 : f64
        affine.store %25, %arg3[%arg7, %arg8] : memref<?x28xf64>
        %26 = affine.load %arg6[%arg8] : memref<?xf64>
        %27 = arith.mulf %11, %26 : f64
        %28 = arith.divf %25, %27 : f64
        affine.store %28, %arg3[%arg7, %arg8] : memref<?x28xf64>
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    %12 = affine.apply #map3()[%1]
    %13 = polygeist.submap(%arg4, %12) {map = #map4} : (memref<?x28xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%13 : memref<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_1 : f64
    }
    %14 = affine.apply #map3()[%1]
    %15 = polygeist.submap(%arg4, %1, %14) {map = #map5} : (memref<?x28xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%15 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      %23 = linalg.index 0 : index
      %24 = linalg.index 1 : index
      %25 = affine.apply #map6(%23)
      %26 = arith.cmpi sge, %24, %25 : index
      %27 = arith.select %26, %cst_0, %out : f64
      linalg.yield %27 : f64
    }
    %16 = affine.apply #map3()[%1]
    %17 = polygeist.submap(%arg3, %0, %1, %16) {map = #map7} : (memref<?x28xf64>, index, index, index) -> memref<?x?x?xf64>
    %18 = polygeist.submap(%arg3, %0, %1, %16) {map = #map8} : (memref<?x28xf64>, index, index, index) -> memref<?x?x?xf64>
    %19 = polygeist.submap(%arg4, %0, %1, %16) {map = #map9} : (memref<?x28xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17, %18 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%19 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %23 = arith.mulf %in, %in_2 : f64
      %24 = arith.addf %out, %23 : f64
      linalg.yield %24 : f64
    }
    %20 = affine.apply #map3()[%1]
    %21 = polygeist.submap(%arg4, %1, %20) {map = #map5} : (memref<?x28xf64>, index, index) -> memref<?x?xf64>
    %22 = polygeist.submap(%arg4, %1, %20) {map = #map1} : (memref<?x28xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%21 : memref<?x?xf64>) outs(%22 : memref<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %23 = linalg.index 0 : index
      %24 = linalg.index 1 : index
      %25 = affine.apply #map6(%23)
      %26 = arith.cmpi sge, %24, %25 : index
      %27 = arith.select %26, %in, %out : f64
      linalg.yield %27 : f64
    }
    affine.store %cst_1, %arg4[symbol(%1) - 1, symbol(%1) - 1] : memref<?x28xf64>
    return
  }
}

