#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x40xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %0 {
      %1 = affine.load %arg3[%arg4] : memref<?xf64>
      affine.store %1, %arg2[%arg4] : memref<?xf64>
      %2 = arith.subi %0, %c1 : index
      %3 = polygeist.submap(%arg1, %arg4, %2) {map = #map} : (memref<?x40xf64>, index, index) -> memref<?xf64>
      %4 = polygeist.submap(%arg2, %2) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
      %5 = polygeist.submap(%arg2, %arg4, %2) {map = #map2} : (memref<?xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["reduction"]} ins(%3, %4 : memref<?xf64>, memref<?xf64>) outs(%5 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %9 = arith.mulf %in, %in_0 : f64
        %10 = arith.subf %out, %9 : f64
        %11 = linalg.index 0 : index
        %12 = arith.cmpi slt, %11, %arg4 : index
        %13 = arith.select %12, %10, %out : f64
        linalg.yield %13 : f64
      }
      %6 = affine.load %arg2[%arg4] : memref<?xf64>
      %7 = affine.load %arg1[%arg4, %arg4] : memref<?x40xf64>
      %8 = arith.divf %6, %7 : f64
      affine.store %8, %arg2[%arg4] : memref<?xf64>
    }
    return
  }
}

