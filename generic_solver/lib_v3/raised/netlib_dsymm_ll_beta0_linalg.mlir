#map = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map4 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ll_beta0(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: i32, %arg8: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg9 = 0 to %1 {
      affine.for %arg10 = 0 to %0 {
        %2 = affine.load %arg6[%arg9, -%arg10 + symbol(%0) - 1] : memref<?x?xf64>
        %3 = arith.mulf %arg2, %2 : f64
        %alloca = memref.alloca() : memref<f64>
        affine.store %cst, %alloca[] : memref<f64>
        %4 = polygeist.submap(%arg4, %arg10, %0, %0) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %5 = polygeist.submap(%arg8, %arg9, %0) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%4 : memref<?xf64>) outs(%5 : memref<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          %14 = arith.mulf %3, %in : f64
          %15 = arith.addf %out, %14 : f64
          %16 = linalg.index 0 : index
          %17 = affine.apply #map3(%arg10)[%0]
          %18 = arith.cmpi sge, %16, %17 : index
          %19 = arith.select %18, %15, %out : f64
          linalg.yield %19 : f64
        }
        %6 = polygeist.submap(%arg6, %arg9, %0) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%arg4, %arg10, %0, %0) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %8 = polygeist.submap(%alloca, %0) {map = #map4} : (memref<f64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction"]} ins(%6, %7 : memref<?xf64>, memref<?xf64>) outs(%8 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %14 = arith.mulf %in, %in_0 : f64
          %15 = arith.addf %out, %14 : f64
          %16 = linalg.index 0 : index
          %17 = affine.apply #map3(%arg10)[%0]
          %18 = arith.cmpi sge, %16, %17 : index
          %19 = arith.select %18, %15, %out : f64
          linalg.yield %19 : f64
        }
        %9 = affine.load %alloca[] : memref<f64>
        %10 = affine.load %arg4[-%arg10 + symbol(%0) - 1, -%arg10 + symbol(%0) - 1] : memref<?x?xf64>
        %11 = arith.mulf %3, %10 : f64
        %12 = arith.mulf %arg2, %9 : f64
        %13 = arith.addf %11, %12 : f64
        affine.store %13, %arg8[%arg9, -%arg10 + symbol(%0) - 1] : memref<?x?xf64>
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

