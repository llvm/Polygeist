#map = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, -d0 - s0 + s1 * 2 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 - s0 + s1 * 2 - 1)>
#map2 = affine_map<(d0)[s0] -> (s0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrsv_ltn(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    %alloca = memref.alloca(%0) : memref<?xf64>
    affine.for %arg4 = 0 to %0 {
      %1 = affine.load %arg3[-%arg4 + symbol(%0) - 1] : memref<?xf64>
      affine.store %1, %alloca[%arg4] : memref<?xf64>
      %2 = polygeist.submap(%arg2, %arg4, %0, %0) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
      %3 = polygeist.submap(%arg3, %arg4, %0, %0) {map = #map1} : (memref<?xf64>, index, index, index) -> memref<?xf64>
      %4 = polygeist.submap(%alloca, %arg4, %0) {map = #map2} : (memref<?xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %3 : memref<?xf64>, memref<?xf64>) outs(%4 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %8 = arith.mulf %in, %in_0 : f64
        %9 = arith.subf %out, %8 : f64
        %10 = linalg.index 0 : index
        %11 = affine.apply #map4(%arg4)[%0]
        %12 = arith.cmpi sge, %10, %11 : index
        %13 = arith.select %12, %9, %out : f64
        linalg.yield %13 : f64
      }
      %5 = affine.load %alloca[%arg4] : memref<?xf64>
      %6 = affine.load %arg2[-%arg4 + symbol(%0) - 1, -%arg4 + symbol(%0) - 1] : memref<?x?xf64>
      %7 = arith.divf %5, %6 : f64
      affine.store %7, %arg3[-%arg4 + symbol(%0) - 1] : memref<?xf64>
    }
    return
  }
}

