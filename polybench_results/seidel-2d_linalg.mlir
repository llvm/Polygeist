#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0 + 1)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0 + 2)>
#map4 = affine_map<(d0, d1, d2) -> (d1 + 1, d0)>
#map5 = affine_map<(d0, d1, d2) -> (d1 + 1, d0 + 2)>
#map6 = affine_map<(d0, d1, d2) -> (d1 + 2, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d1 + 2, d0 + 1)>
#map8 = affine_map<(d0, d1, d2) -> (d1 + 2, d0 + 2)>
#map9 = affine_map<(d0, d1, d2) -> (d1 + 1, d0 + 1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x40xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 9.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = affine.apply #map()[%1]
    %3 = arith.subi %2, %c1 : index
    %4 = affine.apply #map()[%1]
    %5 = arith.subi %4, %c1 : index
    %6 = polygeist.submap(%arg2, %3, %5, %0) {map = #map1} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %7 = affine.apply #map()[%1]
    %8 = arith.subi %7, %c1 : index
    %9 = affine.apply #map()[%1]
    %10 = arith.subi %9, %c1 : index
    %11 = polygeist.submap(%arg2, %8, %10, %0) {map = #map2} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %12 = affine.apply #map()[%1]
    %13 = arith.subi %12, %c1 : index
    %14 = affine.apply #map()[%1]
    %15 = arith.subi %14, %c1 : index
    %16 = polygeist.submap(%arg2, %13, %15, %0) {map = #map3} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %17 = affine.apply #map()[%1]
    %18 = arith.subi %17, %c1 : index
    %19 = affine.apply #map()[%1]
    %20 = arith.subi %19, %c1 : index
    %21 = polygeist.submap(%arg2, %18, %20, %0) {map = #map4} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %22 = affine.apply #map()[%1]
    %23 = arith.subi %22, %c1 : index
    %24 = affine.apply #map()[%1]
    %25 = arith.subi %24, %c1 : index
    %26 = polygeist.submap(%arg2, %23, %25, %0) {map = #map5} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %27 = affine.apply #map()[%1]
    %28 = arith.subi %27, %c1 : index
    %29 = affine.apply #map()[%1]
    %30 = arith.subi %29, %c1 : index
    %31 = polygeist.submap(%arg2, %28, %30, %0) {map = #map6} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %32 = affine.apply #map()[%1]
    %33 = arith.subi %32, %c1 : index
    %34 = affine.apply #map()[%1]
    %35 = arith.subi %34, %c1 : index
    %36 = polygeist.submap(%arg2, %33, %35, %0) {map = #map7} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %37 = affine.apply #map()[%1]
    %38 = arith.subi %37, %c1 : index
    %39 = affine.apply #map()[%1]
    %40 = arith.subi %39, %c1 : index
    %41 = polygeist.submap(%arg2, %38, %40, %0) {map = #map8} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    %42 = affine.apply #map()[%1]
    %43 = arith.subi %42, %c1 : index
    %44 = affine.apply #map()[%1]
    %45 = arith.subi %44, %c1 : index
    %46 = polygeist.submap(%arg2, %43, %45, %0) {map = #map9} : (memref<?x40xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10], iterator_types = ["reduction", "parallel", "parallel"]} ins(%6, %11, %16, %21, %26, %31, %36, %41 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%46 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
      %47 = arith.addf %in, %in_0 : f64
      %48 = arith.addf %47, %in_1 : f64
      %49 = arith.addf %48, %in_2 : f64
      %50 = arith.addf %49, %out : f64
      %51 = arith.addf %50, %in_3 : f64
      %52 = arith.addf %51, %in_4 : f64
      %53 = arith.addf %52, %in_5 : f64
      %54 = arith.addf %53, %in_6 : f64
      %55 = arith.divf %54, %cst : f64
      linalg.yield %55 : f64
    }
    return
  }
}

