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
    %cst = arith.constant 9.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x40xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.apply #map()[%2]
    %4 = arith.subi %3, %c1 : index
    %5 = affine.apply #map()[%2]
    %6 = arith.subi %5, %c1 : index
    %7 = affine.apply #map()[%2]
    %8 = arith.subi %7, %c1 : index
    %9 = affine.apply #map()[%2]
    %10 = arith.subi %9, %c1 : index
    %11 = affine.apply #map()[%2]
    %12 = arith.subi %11, %c1 : index
    %13 = affine.apply #map()[%2]
    %14 = arith.subi %13, %c1 : index
    %15 = affine.apply #map()[%2]
    %16 = arith.subi %15, %c1 : index
    %17 = affine.apply #map()[%2]
    %18 = arith.subi %17, %c1 : index
    %19 = affine.apply #map()[%2]
    %20 = arith.subi %19, %c1 : index
    %21 = affine.apply #map()[%2]
    %22 = arith.subi %21, %c1 : index
    %23 = affine.apply #map()[%2]
    %24 = arith.subi %23, %c1 : index
    %25 = affine.apply #map()[%2]
    %26 = arith.subi %25, %c1 : index
    %27 = affine.apply #map()[%2]
    %28 = arith.subi %27, %c1 : index
    %29 = affine.apply #map()[%2]
    %30 = arith.subi %29, %c1 : index
    %31 = affine.apply #map()[%2]
    %32 = arith.subi %31, %c1 : index
    %33 = affine.apply #map()[%2]
    %34 = arith.subi %33, %c1 : index
    %35 = affine.apply #map()[%2]
    %36 = arith.subi %35, %c1 : index
    %37 = affine.apply #map()[%2]
    %38 = arith.subi %37, %c1 : index
    %39 = polygeist.submap(%0, %4, %6, %1) {map = #map1} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %40 = polygeist.submap(%0, %8, %10, %1) {map = #map2} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %41 = polygeist.submap(%0, %12, %14, %1) {map = #map3} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %42 = polygeist.submap(%0, %16, %18, %1) {map = #map4} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %43 = polygeist.submap(%0, %20, %22, %1) {map = #map5} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %44 = polygeist.submap(%0, %24, %26, %1) {map = #map6} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %45 = polygeist.submap(%0, %28, %30, %1) {map = #map7} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %46 = polygeist.submap(%0, %32, %34, %1) {map = #map8} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %47 = polygeist.submap(%0, %36, %38, %1) {map = #map9} : (tensor<?x40xf64>, index, index, index) -> tensor<?x?x?xf64>
    %48 = linalg.generic {doc = "", indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10], iterator_types = ["reduction", "parallel", "parallel"], library_call = ""} ins(%39, %40, %41, %42, %43, %44, %45, %46 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%47 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
      %51 = arith.addf %in, %in_0 : f64
      %52 = arith.addf %51, %in_1 : f64
      %53 = arith.addf %52, %in_2 : f64
      %54 = arith.addf %53, %out : f64
      %55 = arith.addf %54, %in_3 : f64
      %56 = arith.addf %55, %in_4 : f64
      %57 = arith.addf %56, %in_5 : f64
      %58 = arith.addf %57, %in_6 : f64
      %59 = arith.divf %58, %cst : f64
      linalg.yield %59 : f64
    } -> tensor<?x?x?xf64>
    %49 = polygeist.submapInverse(%0, %48, %36, %38, %1) {map = #map9} : (tensor<?x40xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x40xf64>
    %50 = bufferization.to_memref %49 : memref<?x40xf64>
    memref.copy %50, %arg2 : memref<?x40xf64> to memref<?x40xf64>
    return
  }
}

