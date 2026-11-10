#map = affine_map<(d0)[s0] -> (d0, s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<(d0, d1)[s0] -> (d1, s0)>
#map4 = affine_map<(d0, d1)[s0] -> (d0, s0)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cholesky_right(%arg0: i32, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      %1 = affine.load %arg1[%arg2, %arg2] : memref<?x?xf64>
      %2 = math.sqrt %1 : f64
      affine.store %2, %arg1[%arg2, %arg2] : memref<?x?xf64>
      %3 = polygeist.submap(%arg1, %arg2, %0) {map = #map} : (memref<?x?xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xf64>) {
      ^bb0(%out: f64):
        %7 = arith.divf %out, %2 : f64
        %8 = linalg.index 0 : index
        %9 = affine.apply #map2(%arg2)
        %10 = arith.cmpi sge, %8, %9 : index
        %11 = arith.select %10, %7, %out : f64
        linalg.yield %11 : f64
      }
      %4 = polygeist.submap(%arg1, %arg2, %0, %0) {map = #map3} : (memref<?x?xf64>, index, index, index) -> memref<?x?xf64>
      %5 = polygeist.submap(%arg1, %arg2, %0, %0) {map = #map4} : (memref<?x?xf64>, index, index, index) -> memref<?x?xf64>
      %6 = polygeist.submap(%arg1, %0, %0) {map = #map5} : (memref<?x?xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<?x?xf64>, memref<?x?xf64>) outs(%6 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %7 = linalg.index 0 : index
        %8 = arith.mulf %in, %in_0 : f64
        %9 = arith.subf %out, %8 : f64
        %10 = linalg.index 1 : index
        %11 = arith.cmpi sge, %10, %7 : index
        %12 = arith.select %11, %9, %out : f64
        %13 = linalg.index 0 : index
        %14 = affine.apply #map2(%arg2)
        %15 = arith.cmpi sge, %13, %14 : index
        %16 = arith.select %15, %12, %out : f64
        linalg.yield %16 : f64
      }
    }
    return
  }
}

