#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x30xf64>, %arg5: memref<?x20xf64>, %arg6: memref<?x20xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x20xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?x20xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x30xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.subi %4, %c1 : index
    %6 = affine.apply #map(%5)
    %7 = polygeist.submap(%2, %6, %4) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%7 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %28 = linalg.index 0 : index
      %29 = arith.mulf %out, %arg3 : f64
      %30 = linalg.index 1 : index
      %31 = affine.apply #map(%28)
      %32 = arith.cmpi slt, %30, %31 : index
      %33 = arith.select %32, %29, %out : f64
      linalg.yield %33 : f64
    } -> tensor<?x?xf64>
    %9 = polygeist.submapInverse(%2, %8, %6, %4) {map = #map1} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
    %10 = arith.subi %4, %c1 : index
    %11 = affine.apply #map(%10)
    %12 = arith.subi %4, %c1 : index
    %13 = affine.apply #map(%12)
    %14 = arith.subi %4, %c1 : index
    %15 = affine.apply #map(%14)
    %16 = arith.subi %4, %c1 : index
    %17 = affine.apply #map(%16)
    %18 = arith.subi %4, %c1 : index
    %19 = affine.apply #map(%18)
    %20 = polygeist.submap(%9, %19, %3, %4) {map = #map3} : (tensor<?x30xf64>, index, index, index) -> tensor<?x?x?xf64>
    %21 = polygeist.submap(%1, %11, %3, %4) {map = #map4} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %22 = polygeist.submap(%1, %17, %3, %4) {map = #map5} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %23 = polygeist.submap(%0, %13, %3, %4) {map = #map5} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %24 = polygeist.submap(%0, %15, %3, %4) {map = #map4} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%21, %23, %24, %22 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%20 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %28 = linalg.index 0 : index
      %29 = arith.mulf %in, %arg2 : f64
      %30 = arith.mulf %29, %in_0 : f64
      %31 = arith.mulf %in_1, %arg2 : f64
      %32 = arith.mulf %31, %in_2 : f64
      %33 = arith.addf %30, %32 : f64
      %34 = arith.addf %out, %33 : f64
      %35 = linalg.index 2 : index
      %36 = affine.apply #map(%28)
      %37 = arith.cmpi slt, %35, %36 : index
      %38 = arith.select %37, %34, %out : f64
      linalg.yield %38 : f64
    } -> tensor<?x?x?xf64>
    %26 = polygeist.submapInverse(%9, %25, %19, %3, %4) {map = #map3} : (tensor<?x30xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x30xf64>
    %27 = bufferization.to_memref %26 : memref<?x30xf64>
    memref.copy %27, %arg4 : memref<?x30xf64> to memref<?x30xf64>
    return
  }
}

