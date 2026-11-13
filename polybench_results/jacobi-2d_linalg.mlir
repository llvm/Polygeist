#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d1 + 1, d0 + 1)>
#map2 = affine_map<(d0, d1) -> (d1 + 1, d0)>
#map3 = affine_map<(d0, d1) -> (d1 + 1, d0 + 2)>
#map4 = affine_map<(d0, d1) -> (d1 + 2, d0 + 1)>
#map5 = affine_map<(d0, d1) -> (d1, d0 + 1)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x30xf64>, %arg3: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.000000e-01 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %1 {
      %2 = affine.apply #map()[%0]
      %3 = arith.subi %2, %c1 : index
      %4 = affine.apply #map()[%0]
      %5 = arith.subi %4, %c1 : index
      %6 = polygeist.submap(%arg2, %5, %3) {map = #map1} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %7 = affine.apply #map()[%0]
      %8 = arith.subi %7, %c1 : index
      %9 = polygeist.submap(%arg2, %8, %3) {map = #map2} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %10 = affine.apply #map()[%0]
      %11 = arith.subi %10, %c1 : index
      %12 = polygeist.submap(%arg2, %11, %3) {map = #map3} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %13 = affine.apply #map()[%0]
      %14 = arith.subi %13, %c1 : index
      %15 = polygeist.submap(%arg2, %14, %3) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %16 = affine.apply #map()[%0]
      %17 = arith.subi %16, %c1 : index
      %18 = polygeist.submap(%arg2, %17, %3) {map = #map5} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %19 = affine.apply #map()[%0]
      %20 = arith.subi %19, %c1 : index
      %21 = polygeist.submap(%arg3, %20, %3) {map = #map1} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%6, %9, %12, %15, %18 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%21 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %42 = arith.addf %in, %in_0 : f64
        %43 = arith.addf %42, %in_1 : f64
        %44 = arith.addf %43, %in_2 : f64
        %45 = arith.addf %44, %in_3 : f64
        %46 = arith.mulf %45, %cst : f64
        linalg.yield %46 : f64
      }
      %22 = affine.apply #map()[%0]
      %23 = arith.subi %22, %c1 : index
      %24 = affine.apply #map()[%0]
      %25 = arith.subi %24, %c1 : index
      %26 = polygeist.submap(%arg3, %25, %23) {map = #map1} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %27 = affine.apply #map()[%0]
      %28 = arith.subi %27, %c1 : index
      %29 = polygeist.submap(%arg3, %28, %23) {map = #map2} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %30 = affine.apply #map()[%0]
      %31 = arith.subi %30, %c1 : index
      %32 = polygeist.submap(%arg3, %31, %23) {map = #map3} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %33 = affine.apply #map()[%0]
      %34 = arith.subi %33, %c1 : index
      %35 = polygeist.submap(%arg3, %34, %23) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %36 = affine.apply #map()[%0]
      %37 = arith.subi %36, %c1 : index
      %38 = polygeist.submap(%arg3, %37, %23) {map = #map5} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %39 = affine.apply #map()[%0]
      %40 = arith.subi %39, %c1 : index
      %41 = polygeist.submap(%arg2, %40, %23) {map = #map1} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%26, %29, %32, %35, %38 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%41 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %42 = arith.addf %in, %in_0 : f64
        %43 = arith.addf %42, %in_1 : f64
        %44 = arith.addf %43, %in_2 : f64
        %45 = arith.addf %44, %in_3 : f64
        %46 = arith.mulf %45, %cst : f64
        linalg.yield %46 : f64
      }
    }
    return
  }
}

