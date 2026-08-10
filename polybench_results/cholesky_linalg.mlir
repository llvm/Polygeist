#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0)[s0, s1] -> (s0, s1)>
#map3 = affine_map<(d0)[s0] -> (s0, s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_cholesky(%arg0: i32, %arg1: memref<?x40xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        %6 = arith.subi %arg2, %c1 : index
        %7 = polygeist.submap(%arg1, %arg2, %6) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %8 = polygeist.submap(%arg1, %arg3, %6) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %9 = polygeist.submap(%arg1, %arg2, %arg3, %6) {map = #map2} : (memref<?x40xf64>, index, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%7, %8 : memref<?xf64>, memref<?xf64>) outs(%9 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %13 = arith.mulf %in, %in_0 : f64
          %14 = arith.subf %out, %13 : f64
          %15 = linalg.index 0 : index
          %16 = arith.cmpi slt, %15, %arg3 : index
          %17 = arith.select %16, %14, %out : f64
          linalg.yield %17 : f64
        }
        %10 = affine.load %arg1[%arg3, %arg3] : memref<?x40xf64>
        %11 = affine.load %arg1[%arg2, %arg3] : memref<?x40xf64>
        %12 = arith.divf %11, %10 : f64
        affine.store %12, %arg1[%arg2, %arg3] : memref<?x40xf64>
      }
      %1 = arith.subi %0, %c1 : index
      %2 = polygeist.submap(%arg1, %arg2, %1) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
      %3 = polygeist.submap(%arg1, %arg2, %1) {map = #map3} : (memref<?x40xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["reduction"]} ins(%2 : memref<?xf64>) outs(%3 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %6 = arith.mulf %in, %in : f64
        %7 = arith.subf %out, %6 : f64
        %8 = linalg.index 0 : index
        %9 = arith.cmpi slt, %8, %arg2 : index
        %10 = arith.select %9, %7, %out : f64
        linalg.yield %10 : f64
      }
      %4 = affine.load %arg1[%arg2, %arg2] : memref<?x40xf64>
      %5 = math.sqrt %4 : f64
      affine.store %5, %arg1[%arg2, %arg2] : memref<?x40xf64>
    }
    return
  }
}

