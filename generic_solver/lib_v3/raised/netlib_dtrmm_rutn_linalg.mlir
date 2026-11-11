#map = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0)[s0] -> (s0, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrmm_rutn(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %0, %c1 : index
    affine.for %arg7 = 0 to %0 {
      %3 = polygeist.submap(%arg6, %arg7, %2, %1) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?x?xf64>
      %4 = polygeist.submap(%arg6, %2, %1) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?x?xf64>
      %5 = polygeist.submap(%arg4, %arg7, %2) {map = #map2} : (memref<?x?xf64>, index, index) -> memref<?xf64>
      %6 = polygeist.submap(%5, %2, %1) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %3 : memref<?x?xf64>, memref<?x?xf64>) outs(%4 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %10 = arith.mulf %arg2, %in : f64
        %11 = arith.mulf %10, %in_0 : f64
        %12 = arith.addf %out, %11 : f64
        %13 = linalg.index 0 : index
        %14 = arith.cmpi slt, %13, %arg7 : index
        %15 = arith.select %14, %12, %out : f64
        linalg.yield %15 : f64
      }
      %7 = affine.load %arg4[%arg7, %arg7] : memref<?x?xf64>
      %8 = arith.mulf %arg2, %7 : f64
      %9 = polygeist.submap(%arg6, %arg7, %1) {map = #map2} : (memref<?x?xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%9 : memref<?xf64>) {
      ^bb0(%out: f64):
        %10 = arith.mulf %8, %out : f64
        linalg.yield %10 : f64
      }
    }
    return
  }
}

