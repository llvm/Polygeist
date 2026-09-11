#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_durbin(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<f64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[] : memref<f64>
    %alloca_1 = memref.alloca() : memref<f64>
    affine.store %0, %alloca_1[] : memref<f64>
    %alloca_2 = memref.alloca() : memref<f64>
    affine.store %0, %alloca_2[] : memref<f64>
    %alloca_3 = memref.alloca() : memref<40xf64>
    %1 = affine.load %arg1[0] : memref<?xf64>
    %2 = arith.negf %1 : f64
    affine.store %2, %arg2[0] : memref<?xf64>
    affine.store %cst_0, %alloca_1[] : memref<f64>
    %3 = affine.load %arg1[0] : memref<?xf64>
    %4 = arith.negf %3 : f64
    affine.store %4, %alloca_2[] : memref<f64>
    %5 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 1 to %5 {
      %6 = affine.load %alloca_2[] : memref<f64>
      %7 = arith.mulf %6, %6 : f64
      %8 = arith.subf %cst_0, %7 : f64
      %9 = affine.load %alloca_1[] : memref<f64>
      %10 = arith.mulf %8, %9 : f64
      affine.store %10, %alloca_1[] : memref<f64>
      affine.store %cst, %alloca[] : memref<f64>
      %11 = arith.subi %5, %c1 : index
      %12 = polygeist.submap(%arg1, %arg3, %11) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
      %13 = polygeist.submap(%arg2, %11) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
      %14 = polygeist.submap(%alloca, %11) {map = #map2} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["reduction"]} ins(%12, %13 : memref<?xf64>, memref<?xf64>) outs(%14 : memref<?xf64>) {
      ^bb0(%in: f64, %in_4: f64, %out: f64):
        %27 = arith.mulf %in, %in_4 : f64
        %28 = arith.addf %out, %27 : f64
        %29 = linalg.index 0 : index
        %30 = arith.cmpi slt, %29, %arg3 : index
        %31 = arith.select %30, %28, %out : f64
        linalg.yield %31 : f64
      }
      %15 = affine.load %arg1[%arg3] : memref<?xf64>
      %16 = affine.load %alloca[] : memref<f64>
      %17 = arith.addf %15, %16 : f64
      %18 = arith.negf %17 : f64
      %19 = arith.divf %18, %10 : f64
      affine.store %19, %alloca_2[] : memref<f64>
      %20 = arith.subi %5, %c1 : index
      %21 = polygeist.submap(%arg2, %20) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
      %22 = polygeist.submap(%arg2, %arg3, %20) {map = #map} : (memref<?xf64>, index, index) -> memref<?xf64>
      %23 = polygeist.submap(%alloca_3, %20) {map = #map1} : (memref<40xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%21, %22 : memref<?xf64>, memref<?xf64>) outs(%23 : memref<?xf64>) {
      ^bb0(%in: f64, %in_4: f64, %out: f64):
        %27 = arith.mulf %19, %in_4 : f64
        %28 = arith.addf %in, %27 : f64
        %29 = linalg.index 0 : index
        %30 = arith.cmpi slt, %29, %arg3 : index
        %31 = arith.select %30, %28, %out : f64
        linalg.yield %31 : f64
      }
      %24 = arith.subi %5, %c1 : index
      %25 = polygeist.submap(%alloca_3, %24) {map = #map1} : (memref<40xf64>, index) -> memref<?xf64>
      %26 = polygeist.submap(%arg2, %24) {map = #map1} : (memref<?xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%25 : memref<?xf64>) outs(%26 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %27 = linalg.index 0 : index
        %28 = arith.cmpi slt, %27, %arg3 : index
        %29 = arith.select %28, %in, %out : f64
        linalg.yield %29 : f64
      }
      affine.store %19, %arg2[%arg3] : memref<?xf64>
    }
    return
  }
}

