#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrmm_rutn(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %2, %c1 : index
    %5 = affine.for %arg7 = 0 to %2 iter_args(%arg8 = %0) -> (tensor<?x?xf64>) {
      %7 = polygeist.submap(%1, %arg7, %4) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %8 = polygeist.submap(%7, %4, %3) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %9 = polygeist.submap(%arg8, %arg7, %4, %3) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %10 = polygeist.submap(%arg8, %4, %3) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %11 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%8, %9 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%10 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %17 = arith.mulf %arg2, %in : f64
        %18 = arith.mulf %17, %in_0 : f64
        %19 = arith.addf %out, %18 : f64
        %20 = linalg.index 0 : index
        %21 = arith.cmpi slt, %20, %arg7 : index
        %22 = arith.select %21, %19, %out : f64
        linalg.yield %22 : f64
      } -> tensor<?x?xf64>
      %12 = polygeist.submapInverse(%arg8, %11, %4, %3) {map = #map3} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %extracted = tensor.extract %1[%arg7, %arg7] : tensor<?x?xf64>
      %13 = arith.mulf %arg2, %extracted : f64
      %14 = polygeist.submap(%12, %arg7, %3) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map4], iterator_types = ["parallel"], library_call = ""} outs(%14 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %17 = arith.mulf %13, %out : f64
        linalg.yield %17 : f64
      } -> tensor<?xf64>
      %16 = polygeist.submapInverse(%12, %15, %arg7, %3) {map = #map} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      affine.yield %16 : tensor<?x?xf64>
    }
    %6 = bufferization.to_memref %5 : memref<?x?xf64>
    memref.copy %6, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

