#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0)[s0] -> (d0, s0)>
#map3 = affine_map<(d0)[s0, s1] -> (s0, s1)>
#map4 = affine_map<(d0, d1)[s0] -> (s0, d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1)[s0] -> (s0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_lu(%arg0: i32, %arg1: memref<?x40xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        %7 = arith.subi %arg2, %c1 : index
        %8 = polygeist.submap(%arg1, %arg2, %7) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %9 = polygeist.submap(%arg1, %arg3, %7) {map = #map2} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %10 = polygeist.submap(%arg1, %arg2, %arg3, %7) {map = #map3} : (memref<?x40xf64>, index, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%8, %9 : memref<?xf64>, memref<?xf64>) outs(%10 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %14 = arith.mulf %in, %in_0 : f64
          %15 = arith.subf %out, %14 : f64
          %16 = linalg.index 0 : index
          %17 = arith.cmpi slt, %16, %arg3 : index
          %18 = arith.select %17, %15, %out : f64
          linalg.yield %18 : f64
        }
        %11 = affine.load %arg1[%arg3, %arg3] : memref<?x40xf64>
        %12 = affine.load %arg1[%arg2, %arg3] : memref<?x40xf64>
        %13 = arith.divf %12, %11 : f64
        affine.store %13, %arg1[%arg2, %arg3] : memref<?x40xf64>
      }
      %1 = arith.subi %0, %c1 : index
      %2 = polygeist.submap(%arg1, %arg2, %1, %0) {map = #map4} : (memref<?x40xf64>, index, index, index) -> memref<?x?xf64>
      %3 = arith.subi %0, %c1 : index
      %4 = polygeist.submap(%arg1, %3, %0) {map = #map5} : (memref<?x40xf64>, index, index) -> memref<?x?xf64>
      %5 = arith.subi %0, %c1 : index
      %6 = polygeist.submap(%arg1, %arg2, %5, %0) {map = #map6} : (memref<?x40xf64>, index, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction"]} ins(%2, %4 : memref<?x?xf64>, memref<?x?xf64>) outs(%6 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %7 = arith.mulf %in, %in_0 : f64
        %8 = arith.subf %out, %7 : f64
        %9 = linalg.index 1 : index
        %10 = arith.cmpi slt, %9, %arg2 : index
        %11 = arith.select %10, %8, %out : f64
        linalg.yield %11 : f64
      }
    }
    return
  }
}

