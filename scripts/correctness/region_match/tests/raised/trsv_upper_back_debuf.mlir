#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @trsv_upper_back(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = tensor.empty(%3) : tensor<?xf64>
    %5:2 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %4, %arg6 = %1) -> (tensor<?xf64>, tensor<?xf64>) {
      %7 = affine.apply #map(%arg4)[%3]
      %extracted = tensor.extract %2[%7] : tensor<?xf64>
      %inserted = tensor.insert %extracted into %arg5[%arg4] : tensor<?xf64>
      %8 = polygeist.submap(%0, %arg4, %3, %3) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %9 = polygeist.submap(%arg6, %3) {map = #map2} : (tensor<?xf64>, index) -> tensor<?xf64>
      %10 = polygeist.submap(%inserted, %arg4, %3) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %11 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction"], library_call = ""} ins(%8, %9 : tensor<?xf64>, tensor<?xf64>) outs(%10 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_3: f64, %out: f64):
        %17 = arith.mulf %in, %in_3 : f64
        %18 = arith.subf %out, %17 : f64
        %19 = linalg.index 0 : index
        %20 = affine.apply #map4(%arg4)[%3]
        %21 = arith.cmpi sge, %19, %20 : index
        %22 = arith.select %21, %18, %out : f64
        linalg.yield %22 : f64
      } -> tensor<?xf64>
      %12 = polygeist.submapInverse(%inserted, %11, %arg4, %3) {map = #map3} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
      %extracted_0 = tensor.extract %12[%arg4] : tensor<?xf64>
      %13 = affine.apply #map(%arg4)[%3]
      %14 = affine.apply #map(%arg4)[%3]
      %extracted_1 = tensor.extract %0[%13, %14] : tensor<?x?xf64>
      %15 = arith.divf %extracted_0, %extracted_1 : f64
      %16 = affine.apply #map(%arg4)[%3]
      %inserted_2 = tensor.insert %15 into %arg6[%16] : tensor<?xf64>
      affine.yield %12, %inserted_2 : tensor<?xf64>, tensor<?xf64>
    }
    %6 = bufferization.to_memref %5#1 : memref<?xf64>
    memref.copy %6, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

