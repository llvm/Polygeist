#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<(d0) -> (d0 + 2)>
#map4 = affine_map<(d0) -> (d0 + 3)>
#map5 = affine_map<(d0) -> (d0 + 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_ddot(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.remsi %0, %c5 : index
    %2 = arith.addi %1, %c5 : index
    %3 = arith.cmpi slt, %1, %c0 : index
    %4 = arith.select %3, %2, %1 : index
    %5 = llvm.mlir.undef : f64
    %6 = arith.remsi %arg0, %c5_i32 : i32
    %7 = arith.cmpi ne, %6, %c0_i32 : i32
    %8:3 = scf.if %7 -> (f64, i1, f64) {
      %alloca = memref.alloca() : memref<f64>
      affine.store %cst, %alloca[] : memref<f64>
      %11 = polygeist.submap(%arg1, %4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
      %12 = polygeist.submap(%arg2, %4) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
      %13 = polygeist.submap(%alloca, %4) {map = #map1} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%11, %12 : memref<?xf64>, memref<?xf64>) outs(%13 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %18 = arith.mulf %in, %in_0 : f64
        %19 = arith.addf %out, %18 : f64
        linalg.yield %19 : f64
      }
      %14 = affine.load %alloca[] : memref<f64>
      %15 = arith.cmpi slt, %arg0, %c5_i32 : i32
      %16 = arith.cmpi sge, %arg0, %c5_i32 : i32
      %17 = arith.select %15, %14, %5 : f64
      scf.yield %14, %16, %17 : f64, i1, f64
    } else {
      scf.yield %cst, %true, %5 : f64, i1, f64
    }
    %9 = scf.if %8#1 -> (f64) {
      %alloca = memref.alloca() : memref<f64>
      affine.store %8#0, %alloca[] : memref<f64>
      %11 = polygeist.submap(%arg1, %0) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
      %12 = polygeist.submap(%arg2, %0) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
      %13 = polygeist.submap(%arg1, %0) {map = #map2} : (memref<?xf64>, index) -> memref<?xf64>
      %14 = polygeist.submap(%arg2, %0) {map = #map2} : (memref<?xf64>, index) -> memref<?xf64>
      %15 = polygeist.submap(%arg1, %0) {map = #map3} : (memref<?xf64>, index) -> memref<?xf64>
      %16 = polygeist.submap(%arg2, %0) {map = #map3} : (memref<?xf64>, index) -> memref<?xf64>
      %17 = polygeist.submap(%arg1, %0) {map = #map4} : (memref<?xf64>, index) -> memref<?xf64>
      %18 = polygeist.submap(%arg2, %0) {map = #map4} : (memref<?xf64>, index) -> memref<?xf64>
      %19 = polygeist.submap(%arg1, %0) {map = #map5} : (memref<?xf64>, index) -> memref<?xf64>
      %20 = polygeist.submap(%arg2, %0) {map = #map5} : (memref<?xf64>, index) -> memref<?xf64>
      %21 = polygeist.submap(%alloca, %0) {map = #map1} : (memref<f64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map, #map, #map, #map, #map, #map, #map, #map, #map, #map, #map], iterator_types = ["reduction"]} ins(%11, %12, %13, %14, %15, %16, %17, %18, %19, %20 : memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%21 : memref<?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
        %23 = arith.mulf %in, %in_0 : f64
        %24 = arith.addf %out, %23 : f64
        %25 = arith.mulf %in_1, %in_2 : f64
        %26 = arith.addf %24, %25 : f64
        %27 = arith.mulf %in_3, %in_4 : f64
        %28 = arith.addf %26, %27 : f64
        %29 = arith.mulf %in_5, %in_6 : f64
        %30 = arith.addf %28, %29 : f64
        %31 = arith.mulf %in_7, %in_8 : f64
        %32 = arith.addf %30, %31 : f64
        %33 = linalg.index 0 : index
        %34 = arith.cmpi sge, %33, %4 : index
        %35 = arith.select %34, %32, %out : f64
        linalg.yield %35 : f64
      }
      %22 = affine.load %alloca[] : memref<f64>
      scf.yield %22 : f64
    } else {
      scf.yield %8#0 : f64
    }
    %10 = arith.select %8#1, %9, %8#2 : f64
    return %10 : f64
  }
}

