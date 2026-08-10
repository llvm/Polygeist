#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x30xf64>, %arg5: memref<?x20xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg5 : memref<?x20xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x30xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %5 = affine.apply #map(%4)
    %6 = polygeist.submap(%1, %5, %3) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?x?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%6 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %21 = linalg.index 0 : index
      %22 = arith.mulf %out, %arg3 : f64
      %23 = linalg.index 1 : index
      %24 = affine.apply #map(%21)
      %25 = arith.cmpi slt, %23, %24 : index
      %26 = arith.select %25, %22, %out : f64
      linalg.yield %26 : f64
    } -> tensor<?x?xf64>
    %8 = polygeist.submapInverse(%1, %7, %5, %3) {map = #map1} : (tensor<?x30xf64>, tensor<?x?xf64>, index, index) -> tensor<?x30xf64>
    %9 = arith.subi %3, %c1 : index
    %10 = affine.apply #map(%9)
    %11 = arith.subi %3, %c1 : index
    %12 = affine.apply #map(%11)
    %13 = arith.subi %3, %c1 : index
    %14 = affine.apply #map(%13)
    %15 = polygeist.submap(%8, %14, %2, %3) {map = #map3} : (tensor<?x30xf64>, index, index, index) -> tensor<?x?x?xf64>
    %16 = polygeist.submap(%0, %10, %2, %3) {map = #map4} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %17 = polygeist.submap(%0, %12, %2, %3) {map = #map5} : (tensor<?x20xf64>, index, index, index) -> tensor<?x?x?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%16, %17 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%15 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %21 = linalg.index 0 : index
      %22 = arith.mulf %arg2, %in : f64
      %23 = arith.mulf %22, %in_0 : f64
      %24 = arith.addf %out, %23 : f64
      %25 = linalg.index 2 : index
      %26 = affine.apply #map(%21)
      %27 = arith.cmpi slt, %25, %26 : index
      %28 = arith.select %27, %24, %out : f64
      linalg.yield %28 : f64
    } -> tensor<?x?x?xf64>
    %19 = polygeist.submapInverse(%8, %18, %14, %2, %3) {map = #map3} : (tensor<?x30xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x30xf64>
    %20 = bufferization.to_memref %19 : memref<?x30xf64>
    memref.copy %20, %arg4 : memref<?x30xf64> to memref<?x30xf64>
    return
  }
}

