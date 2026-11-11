#map = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (-s0 + s1 - 1, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map5 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map6 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrmm_rltn(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.for %arg7 = 0 to %2 iter_args(%arg8 = %1) -> (tensor<?x?xf64>) {
      %6 = polygeist.submap(%0, %arg7, %2, %2) {map = #map} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %7 = polygeist.submap(%6, %2, %3) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
      %8 = polygeist.submap(%arg8, %arg7, %2, %2, %3) {map = #map2} : (tensor<?x?xf64>, index, index, index, index) -> tensor<?x?xf64>
      %9 = polygeist.submap(%arg8, %2, %3) {map = #map3} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %10 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%7, %8 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%9 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %18 = arith.mulf %arg2, %in : f64
        %19 = arith.mulf %18, %in_0 : f64
        %20 = arith.addf %out, %19 : f64
        %21 = linalg.index 0 : index
        %22 = affine.apply #map4(%arg7)[%2]
        %23 = arith.cmpi sge, %21, %22 : index
        %24 = arith.select %23, %20, %out : f64
        linalg.yield %24 : f64
      } -> tensor<?x?xf64>
      %11 = polygeist.submapInverse(%arg8, %10, %2, %3) {map = #map3} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
      %12 = affine.apply #map5(%arg7)[%2]
      %13 = affine.apply #map5(%arg7)[%2]
      %extracted = tensor.extract %0[%12, %13] : tensor<?x?xf64>
      %14 = arith.mulf %arg2, %extracted : f64
      %15 = polygeist.submap(%11, %arg7, %2, %3) {map = #map} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %16 = linalg.generic {doc = "", indexing_maps = [#map6], iterator_types = ["parallel"], library_call = ""} outs(%15 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %18 = arith.mulf %14, %out : f64
        linalg.yield %18 : f64
      } -> tensor<?xf64>
      %17 = polygeist.submapInverse(%11, %16, %arg7, %2, %3) {map = #map} : (tensor<?x?xf64>, tensor<?xf64>, index, index, index) -> tensor<?x?xf64>
      affine.yield %17 : tensor<?x?xf64>
    }
    %5 = bufferization.to_memref %4 : memref<?x?xf64>
    memref.copy %5, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

