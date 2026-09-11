#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2 + 1)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2 + 2)>
#map4 = affine_map<(d0, d1, d2) -> (d1 + 1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1 + 1, d2 + 2)>
#map6 = affine_map<(d0, d1, d2) -> (d1 + 2, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d1 + 2, d2 + 1)>
#map8 = affine_map<(d0, d1, d2) -> (d1 + 2, d2 + 2)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.apply #map()[%2]
    %4 = arith.subi %3, %c1 : index
    %5 = affine.apply #map()[%2]
    %6 = arith.subi %5, %c1 : index
    %7 = polygeist.submap(%0, %1, %6, %4) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %8 = polygeist.submap(%0, %1, %6, %4) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %9 = polygeist.submap(%0, %1, %6, %4) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %10 = polygeist.submap(%0, %1, %6, %4) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %11 = polygeist.submap(%0, %1, %6, %4) {map = #map5} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %12 = polygeist.submap(%0, %1, %6, %4) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %13 = polygeist.submap(%0, %1, %6, %4) {map = #map7} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %14 = polygeist.submap(%0, %1, %6, %4) {map = #map8} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %extracted_slice = tensor.extract_slice %0[1, 1] [%6, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map9, #map9, #map9, #map9, #map9, #map9, #map9, #map9, #map1], iterator_types = ["reduction", "parallel", "parallel"], library_call = ""} ins(%7, %8, %9, %10, %11, %12, %13, %14 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%extracted_slice : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
      %17 = arith.addf %in, %in_0 : f64
      %18 = arith.addf %17, %in_1 : f64
      %19 = arith.addf %18, %in_2 : f64
      %20 = arith.addf %19, %out : f64
      %21 = arith.addf %20, %in_3 : f64
      %22 = arith.addf %21, %in_4 : f64
      %23 = arith.addf %22, %in_5 : f64
      %24 = arith.addf %23, %in_6 : f64
      %25 = arith.divf %24, %cst : f64
      linalg.yield %25 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %15 into %0[1, 1] [%6, %4] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %16 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %16, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

