#map = affine_map<(d0)[s0] -> (d0, s0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (s0, s0)>
#map4 = affine_map<(d0)[s0] -> (s0, d0)>
#map5 = affine_map<(d0) -> (d0 + 1)>
#map6 = affine_map<(d0, d1)[s0] -> (d0, s0)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1)[s0] -> (s0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gramschmidt(%arg0: i32, %arg1: i32, %arg2: memref<?x30xf64>, %arg3: memref<?x30xf64>, %arg4: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg5 = 0 to %0 {
      %alloca = memref.alloca() : memref<f64>
      affine.store %cst, %alloca[] : memref<f64>
      %2 = polygeist.submap(%arg2, %arg5, %1) {map = #map} : (memref<?x30xf64>, index, index) -> memref<?xf64>
      %3 = polygeist.submap(%alloca, %1) {map = #map1} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["reduction"]} ins(%2 : memref<?xf64>) outs(%3 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %16 = arith.mulf %in, %in : f64
        %17 = arith.addf %out, %16 : f64
        linalg.yield %17 : f64
      }
      %4 = affine.load %alloca[] : memref<f64>
      %5 = math.sqrt %4 : f64
      affine.store %5, %arg3[%arg5, %arg5] : memref<?x30xf64>
      %6 = polygeist.submap(%arg2, %arg5, %1) {map = #map} : (memref<?x30xf64>, index, index) -> memref<?xf64>
      %7 = polygeist.submap(%arg3, %arg5, %1) {map = #map3} : (memref<?x30xf64>, index, index) -> memref<?xf64>
      %8 = polygeist.submap(%arg4, %arg5, %1) {map = #map} : (memref<?x30xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%6, %7 : memref<?xf64>, memref<?xf64>) outs(%8 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %16 = arith.divf %in, %in_0 : f64
        linalg.yield %16 : f64
      }
      %9 = polygeist.submap(%arg3, %arg5, %0) {map = #map4} : (memref<?x30xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%9 : memref<?xf64>) {
      ^bb0(%out: f64):
        %16 = linalg.index 0 : index
        %17 = affine.apply #map5(%arg5)
        %18 = arith.cmpi sge, %16, %17 : index
        %19 = arith.select %18, %cst, %out : f64
        linalg.yield %19 : f64
      }
      %10 = polygeist.submap(%arg4, %arg5, %1, %0) {map = #map6} : (memref<?x30xf64>, index, index, index) -> memref<?x?xf64>
      %11 = polygeist.submap(%arg2, %1, %0) {map = #map7} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      %12 = polygeist.submap(%arg3, %arg5, %1, %0) {map = #map8} : (memref<?x30xf64>, index, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "reduction"]} ins(%10, %11 : memref<?x?xf64>, memref<?x?xf64>) outs(%12 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %16 = arith.mulf %in, %in_0 : f64
        %17 = arith.addf %out, %16 : f64
        linalg.yield %17 : f64
      }
      %13 = polygeist.submap(%arg4, %arg5, %1, %0) {map = #map6} : (memref<?x30xf64>, index, index, index) -> memref<?x?xf64>
      %14 = polygeist.submap(%arg3, %arg5, %1, %0) {map = #map8} : (memref<?x30xf64>, index, index, index) -> memref<?x?xf64>
      %15 = polygeist.submap(%arg2, %1, %0) {map = #map7} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%13, %14 : memref<?x?xf64>, memref<?x?xf64>) outs(%15 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %16 = arith.mulf %in, %in_0 : f64
        %17 = arith.subf %out, %16 : f64
        linalg.yield %17 : f64
      }
    }
    return
  }
}

