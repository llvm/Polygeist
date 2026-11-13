#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyrk_un_beta0(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.subi %0, %c1 : index
    %3 = affine.apply #map(%2)
    %4 = polygeist.submap(%arg6, %0, %3) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%4 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      %10 = linalg.index 0 : index
      %11 = linalg.index 1 : index
      %12 = affine.apply #map(%10)
      %13 = arith.cmpi slt, %11, %12 : index
      %14 = arith.select %13, %cst, %out : f64
      linalg.yield %14 : f64
    }
    %5 = arith.subi %0, %c1 : index
    %6 = affine.apply #map(%5)
    %7 = polygeist.submap(%arg4, %0, %1, %6) {map = #map2} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    %8 = polygeist.submap(%arg4, %0, %1, %6) {map = #map3} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    %9 = polygeist.submap(%arg6, %0, %1, %6) {map = #map4} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%7, %8 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%9 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %10 = linalg.index 0 : index
      %11 = arith.mulf %arg2, %in : f64
      %12 = arith.mulf %11, %in_0 : f64
      %13 = arith.addf %out, %12 : f64
      %14 = linalg.index 2 : index
      %15 = affine.apply #map(%10)
      %16 = arith.cmpi slt, %14, %15 : index
      %17 = arith.select %16, %13, %out : f64
      linalg.yield %17 : f64
    }
    return
  }
}

