#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x25xf64>, %arg6: memref<?x30xf64>, %arg7: memref<?x25xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = polygeist.submap(%arg5, %0, %2) {map = #map} : (memref<?x25xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%3 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      %7 = arith.mulf %out, %arg4 : f64
      linalg.yield %7 : f64
    }
    %4 = polygeist.submap(%arg6, %0, %1, %2) {map = #map2} : (memref<?x30xf64>, index, index, index) -> memref<?x?x?xf64>
    %5 = polygeist.submap(%arg7, %0, %1, %2) {map = #map3} : (memref<?x25xf64>, index, index, index) -> memref<?x?x?xf64>
    %6 = polygeist.submap(%arg5, %0, %1, %2) {map = #map4} : (memref<?x25xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4, %5 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%6 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %7 = arith.mulf %arg3, %in : f64
      %8 = arith.mulf %7, %in_0 : f64
      %9 = arith.addf %out, %8 : f64
      linalg.yield %9 : f64
    }
    return
  }
}

