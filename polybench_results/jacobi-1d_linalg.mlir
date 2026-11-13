#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<(d0) -> (d0 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 3.333300e-01 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %1 {
      %2 = affine.apply #map()[%0]
      %3 = arith.subi %2, %c1 : index
      %4 = polygeist.submap(%arg2, %3) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
      %5 = polygeist.submap(%arg2, %3) {map = #map2} : (memref<?xf64>, index) -> memref<?xf64>
      %6 = polygeist.submap(%arg2, %3) {map = #map3} : (memref<?xf64>, index) -> memref<?xf64>
      %7 = polygeist.submap(%arg3, %3) {map = #map2} : (memref<?xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%4, %5, %6 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%7 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
        %14 = arith.addf %in, %in_0 : f64
        %15 = arith.addf %14, %in_1 : f64
        %16 = arith.mulf %15, %cst : f64
        linalg.yield %16 : f64
      }
      %8 = affine.apply #map()[%0]
      %9 = arith.subi %8, %c1 : index
      %10 = polygeist.submap(%arg3, %9) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
      %11 = polygeist.submap(%arg3, %9) {map = #map2} : (memref<?xf64>, index) -> memref<?xf64>
      %12 = polygeist.submap(%arg3, %9) {map = #map3} : (memref<?xf64>, index) -> memref<?xf64>
      %13 = polygeist.submap(%arg2, %9) {map = #map2} : (memref<?xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%10, %11, %12 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%13 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
        %14 = arith.addf %in, %in_0 : f64
        %15 = arith.addf %14, %in_1 : f64
        %16 = arith.mulf %15, %cst : f64
        linalg.yield %16 : f64
      }
    }
    return
  }
}

