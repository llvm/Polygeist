#map = affine_map<(d0)[s0] -> (s0)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1 + 1, d0)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0 + 1)>
#map7 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x30xf64>, %arg4: memref<?x30xf64>, %arg5: memref<?x30xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.69999999999999996 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      %3 = polygeist.submap(%arg6, %arg7, %1) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
      %4 = polygeist.submap(%arg4, %1) {map = #map1} : (memref<?x30xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%3 : memref<?xf64>) outs(%4 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      }
      %5 = arith.subi %0, %c1 : index
      %6 = polygeist.submap(%arg5, %1, %5) {map = #map3} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %7 = polygeist.submap(%arg5, %1, %5) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %8 = polygeist.submap(%arg4, %1, %5) {map = #map3} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%6, %7 : memref<?x?xf64>, memref<?x?xf64>) outs(%8 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %26 = arith.subf %in, %in_1 : f64
        %27 = arith.mulf %26, %cst : f64
        %28 = arith.subf %out, %27 : f64
        linalg.yield %28 : f64
      }
      %9 = arith.subi %1, %c1 : index
      %10 = polygeist.submap(%arg5, %9, %0) {map = #map6} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %11 = arith.subi %1, %c1 : index
      %12 = polygeist.submap(%arg5, %11, %0) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %13 = arith.subi %1, %c1 : index
      %14 = polygeist.submap(%arg3, %13, %0) {map = #map6} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%10, %12 : memref<?x?xf64>, memref<?x?xf64>) outs(%14 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %out: f64):
        %26 = arith.subf %in, %in_1 : f64
        %27 = arith.mulf %26, %cst : f64
        %28 = arith.subf %out, %27 : f64
        linalg.yield %28 : f64
      }
      %15 = affine.apply #map7()[%0]
      %16 = affine.apply #map7()[%1]
      %17 = polygeist.submap(%arg3, %16, %15) {map = #map6} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %18 = affine.apply #map7()[%1]
      %19 = polygeist.submap(%arg3, %18, %15) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %20 = affine.apply #map7()[%1]
      %21 = polygeist.submap(%arg4, %20, %15) {map = #map3} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %22 = affine.apply #map7()[%1]
      %23 = polygeist.submap(%arg4, %22, %15) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %24 = affine.apply #map7()[%1]
      %25 = polygeist.submap(%arg5, %24, %15) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%17, %19, %21, %23 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%25 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
        %26 = arith.subf %in, %in_1 : f64
        %27 = arith.addf %26, %in_2 : f64
        %28 = arith.subf %27, %in_3 : f64
        %29 = arith.mulf %28, %cst_0 : f64
        %30 = arith.subf %out, %29 : f64
        linalg.yield %30 : f64
      }
    }
    return
  }
}

