#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, -d0 - s0 + s1 - 2)>
#map2 = affine_map<(d0)[s0, s1] -> (-d0 - s0 + s1 - 2)>
#map3 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrsv_unn(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = affine.apply #map(%c0)[%0]
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg3[-%arg4 + symbol(%0) - 1] : memref<?xf64>
      %3 = affine.load %arg2[-%arg4 + symbol(%0) - 1, -%arg4 + symbol(%0) - 1] : memref<?x?xf64>
      %4 = arith.divf %2, %3 : f64
      affine.store %4, %arg3[-%arg4 + symbol(%0) - 1] : memref<?xf64>
      %5 = polygeist.submap(%arg2, %arg4, %0, %1) {map = #map1} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
      %6 = polygeist.submap(%arg3, %arg4, %0, %1) {map = #map2} : (memref<?xf64>, index, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%5 : memref<?xf64>) outs(%6 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %7 = arith.mulf %4, %in : f64
        %8 = arith.subf %out, %7 : f64
        %9 = linalg.index 0 : index
        %10 = affine.apply #map(%arg4)[%0]
        %11 = arith.cmpi slt, %9, %10 : index
        %12 = arith.select %11, %8, %out : f64
        linalg.yield %12 : f64
      }
    }
    return
  }
}

