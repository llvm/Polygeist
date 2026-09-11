#map = affine_map<(d0, d1)[s0] -> (d0, s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0)[s0] -> (s0, d0)>
#map5 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x20xf64>, %arg4: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg5 = 0 to %1 {
      %2 = polygeist.submap(%arg3, %arg5, %1, %0) {map = #map} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %3 = polygeist.submap(%arg4, %1, %0) {map = #map1} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %4 = polygeist.submap(%arg4, %arg5, %1, %0) {map = #map2} : (memref<?x30xf64>, index, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%2, %3 : memref<?x?xf64>, memref<?x?xf64>) outs(%4 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %6 = arith.mulf %in, %in_0 : f64
        %7 = arith.addf %out, %6 : f64
        %8 = linalg.index 1 : index
        %9 = affine.apply #map3(%arg5)
        %10 = arith.cmpi sge, %8, %9 : index
        %11 = arith.select %10, %7, %out : f64
        linalg.yield %11 : f64
      }
      %5 = polygeist.submap(%arg4, %arg5, %0) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel"]} outs(%5 : memref<?xf64>) {
      ^bb0(%out: f64):
        %6 = arith.mulf %arg2, %out : f64
        linalg.yield %6 : f64
      }
    }
    return
  }
}

