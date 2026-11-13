#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0)[s0] -> (d0, s0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map5 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_ludcmp(%arg0: i32, %arg1: memref<?x40xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %alloca = memref.alloca() : memref<f64>
    %1 = llvm.mlir.undef : f64
    affine.store %1, %alloca[] : memref<f64>
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to #map(%arg5) {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<?x40xf64>
        affine.store %2, %alloca[] : memref<f64>
        %3 = arith.subi %arg5, %c1 : index
        %4 = polygeist.submap(%arg1, %arg5, %3) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %5 = polygeist.submap(%arg1, %arg6, %3) {map = #map2} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %6 = polygeist.submap(%alloca, %3) {map = #map3} : (memref<f64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%4, %5 : memref<?xf64>, memref<?xf64>) outs(%6 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %10 = arith.mulf %in, %in_0 : f64
          %11 = arith.subf %out, %10 : f64
          %12 = linalg.index 0 : index
          %13 = arith.cmpi slt, %12, %arg6 : index
          %14 = arith.select %13, %11, %out : f64
          linalg.yield %14 : f64
        }
        %7 = affine.load %alloca[] : memref<f64>
        %8 = affine.load %arg1[%arg6, %arg6] : memref<?x40xf64>
        %9 = arith.divf %7, %8 : f64
        affine.store %9, %arg1[%arg5, %arg6] : memref<?x40xf64>
      }
      affine.for %arg6 = #map(%arg5) to %0 {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<?x40xf64>
        affine.store %2, %alloca[] : memref<f64>
        %3 = arith.subi %0, %c1 : index
        %4 = polygeist.submap(%arg1, %arg5, %3) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %5 = polygeist.submap(%arg1, %arg6, %3) {map = #map2} : (memref<?x40xf64>, index, index) -> memref<?xf64>
        %6 = polygeist.submap(%alloca, %3) {map = #map3} : (memref<f64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%4, %5 : memref<?xf64>, memref<?xf64>) outs(%6 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %8 = arith.mulf %in, %in_0 : f64
          %9 = arith.subf %out, %8 : f64
          %10 = linalg.index 0 : index
          %11 = arith.cmpi slt, %10, %arg5 : index
          %12 = arith.select %11, %9, %out : f64
          linalg.yield %12 : f64
        }
        %7 = affine.load %alloca[] : memref<f64>
        affine.store %7, %arg1[%arg5, %arg6] : memref<?x40xf64>
      }
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg2[%arg5] : memref<?xf64>
      affine.store %2, %alloca[] : memref<f64>
      %3 = arith.subi %0, %c1 : index
      %4 = polygeist.submap(%arg1, %arg5, %3) {map = #map1} : (memref<?x40xf64>, index, index) -> memref<?xf64>
      %5 = polygeist.submap(%arg4, %3) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
      %6 = polygeist.submap(%alloca, %3) {map = #map3} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%4, %5 : memref<?xf64>, memref<?xf64>) outs(%6 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %8 = arith.mulf %in, %in_0 : f64
        %9 = arith.subf %out, %8 : f64
        %10 = linalg.index 0 : index
        %11 = arith.cmpi slt, %10, %arg5 : index
        %12 = arith.select %11, %9, %out : f64
        linalg.yield %12 : f64
      }
      %7 = affine.load %alloca[] : memref<f64>
      affine.store %7, %arg4[%arg5] : memref<?xf64>
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg4[-%arg5 + symbol(%0) - 1] : memref<?xf64>
      affine.store %2, %alloca[] : memref<f64>
      %3 = polygeist.submap(%arg1, %arg5, %0, %0) {map = #map4} : (memref<?x40xf64>, index, index, index) -> memref<?xf64>
      %4 = polygeist.submap(%arg3, %0) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
      %5 = polygeist.submap(%alloca, %0) {map = #map3} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%3, %4 : memref<?xf64>, memref<?xf64>) outs(%5 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %9 = arith.mulf %in, %in_0 : f64
        %10 = arith.subf %out, %9 : f64
        %11 = linalg.index 0 : index
        %12 = affine.apply #map5(%arg5)[%0]
        %13 = arith.cmpi sge, %11, %12 : index
        %14 = arith.select %13, %10, %out : f64
        linalg.yield %14 : f64
      }
      %6 = affine.load %alloca[] : memref<f64>
      %7 = affine.load %arg1[-%arg5 + symbol(%0) - 1, -%arg5 + symbol(%0) - 1] : memref<?x40xf64>
      %8 = arith.divf %6, %7 : f64
      affine.store %8, %arg3[-%arg5 + symbol(%0) - 1] : memref<?xf64>
    }
    return
  }
}

