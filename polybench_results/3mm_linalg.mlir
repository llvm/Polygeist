#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x18xf64>, %arg6: memref<?x20xf64>, %arg7: memref<?x18xf64>, %arg8: memref<?x22xf64>, %arg9: memref<?x24xf64>, %arg10: memref<?x22xf64>, %arg11: memref<?x22xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = polygeist.submap(%arg5, %0, %4) {map = #map} : (memref<?x18xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%5 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %6 = polygeist.submap(%arg6, %1, %0, %4) {map = #map2} : (memref<?x20xf64>, index, index, index) -> memref<?x?x?xf64>
    %7 = polygeist.submap(%arg7, %1, %0, %4) {map = #map3} : (memref<?x18xf64>, index, index, index) -> memref<?x?x?xf64>
    %8 = polygeist.submap(%arg5, %1, %0, %4) {map = #map4} : (memref<?x18xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%8 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = arith.mulf %in, %in_0 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    %9 = polygeist.submap(%arg8, %3, %0) {map = #map} : (memref<?x22xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%9 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %10 = polygeist.submap(%arg9, %2, %3, %0) {map = #map2} : (memref<?x24xf64>, index, index, index) -> memref<?x?x?xf64>
    %11 = polygeist.submap(%arg10, %2, %3, %0) {map = #map3} : (memref<?x22xf64>, index, index, index) -> memref<?x?x?xf64>
    %12 = polygeist.submap(%arg8, %2, %3, %0) {map = #map4} : (memref<?x22xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %11 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%12 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = arith.mulf %in, %in_0 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    %13 = polygeist.submap(%arg11, %3, %4) {map = #map} : (memref<?x22xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%13 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %14 = polygeist.submap(%arg5, %0, %3, %4) {map = #map2} : (memref<?x18xf64>, index, index, index) -> memref<?x?x?xf64>
    %15 = polygeist.submap(%arg8, %0, %3, %4) {map = #map3} : (memref<?x22xf64>, index, index, index) -> memref<?x?x?xf64>
    %16 = polygeist.submap(%arg11, %0, %3, %4) {map = #map4} : (memref<?x22xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14, %15 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%16 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = arith.mulf %in, %in_0 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    }
    return
  }
}

