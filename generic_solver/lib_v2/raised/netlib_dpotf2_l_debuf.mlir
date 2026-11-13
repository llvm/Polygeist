#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (d0, s0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0 + 1)>
#map5 = affine_map<(d0, d1)[s0] -> (s0, d1 + s0 + 1)>
#map6 = affine_map<(d0, d1) -> (d0)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0)[s0] -> (s0, d0 + s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dpotf2_l(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant -1.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.subi %0, %c1 : index
    %2 = arith.subi %0, %c1 : index
    %3 = affine.apply #map(%2)[%0]
    %4 = arith.subi %0, %c1 : index
    %5 = arith.subi %0, %c1 : index
    %6 = affine.apply #map(%5)[%0]
    affine.for %arg3 = 0 to %0 {
      %alloca = memref.alloca() : memref<f64>
      affine.store %cst, %alloca[] : memref<f64>
      %7 = polygeist.submap(%arg2, %arg3, %1) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?xf64>
      %8 = polygeist.submap(%alloca, %1) {map = #map2} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["reduction"]} ins(%7 : memref<?xf64>) outs(%8 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %19 = arith.mulf %in, %in : f64
        %20 = arith.addf %out, %19 : f64
        %21 = linalg.index 0 : index
        %22 = arith.cmpi slt, %21, %arg3 : index
        %23 = arith.select %22, %20, %out : f64
        linalg.yield %23 : f64
      }
      %9 = affine.load %alloca[] : memref<f64>
      %10 = affine.load %arg2[%arg3, %arg3] : memref<?x?xf64>
      %11 = arith.subf %10, %9 : f64
      %12 = math.sqrt %11 : f64
      affine.store %12, %arg2[%arg3, %arg3] : memref<?x?xf64>
      %13 = polygeist.submap(%arg2, %arg3, %4, %3) {map = #map4} : (memref<?x?xf64>, index, index, index) -> memref<?x?xf64>
      %14 = polygeist.submap(%arg2, %arg3, %4, %3) {map = #map5} : (memref<?x?xf64>, index, index, index) -> memref<?x?xf64>
      %15 = polygeist.submap(%arg2, %arg3, %4) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?xf64>
      %16 = polygeist.submap(%15, %4, %3) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["reduction", "parallel"]} ins(%16, %13 : memref<?x?xf64>, memref<?x?xf64>) outs(%14 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %out: f64):
        %19 = arith.mulf %in, %cst_0 : f64
        %20 = arith.mulf %19, %in_2 : f64
        %21 = arith.addf %out, %20 : f64
        %22 = linalg.index 1 : index
        %23 = affine.apply #map(%arg3)[%0]
        %24 = arith.cmpi slt, %22, %23 : index
        %25 = arith.select %24, %21, %out : f64
        %26 = linalg.index 0 : index
        %27 = arith.cmpi slt, %26, %arg3 : index
        %28 = arith.select %27, %25, %out : f64
        linalg.yield %28 : f64
      }
      %17 = arith.divf %cst_1, %12 : f64
      %18 = polygeist.submap(%arg2, %arg3, %6) {map = #map8} : (memref<?x?xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%18 : memref<?xf64>) {
      ^bb0(%out: f64):
        %19 = arith.mulf %17, %out : f64
        %20 = linalg.index 0 : index
        %21 = affine.apply #map(%arg3)[%0]
        %22 = arith.cmpi slt, %20, %21 : index
        %23 = arith.select %22, %19, %out : f64
        linalg.yield %23 : f64
      }
    }
    return
  }
}

