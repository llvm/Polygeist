#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, -d0 - s0 + s1 - 2)>
#map2 = affine_map<(d0)[s0, s1] -> (-d0 - s0 + s1 - 2)>
#map3 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrsv_unn(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = affine.apply #map(%c0)[%2]
    %4 = affine.for %arg4 = 0 to %2 iter_args(%arg5 = %1) -> (tensor<?xf64>) {
      %6 = affine.apply #map(%arg4)[%2]
      %extracted = tensor.extract %arg5[%6] : tensor<?xf64>
      %7 = affine.apply #map(%arg4)[%2]
      %8 = affine.apply #map(%arg4)[%2]
      %extracted_0 = tensor.extract %0[%7, %8] : tensor<?x?xf64>
      %9 = arith.divf %extracted, %extracted_0 : f64
      %10 = affine.apply #map(%arg4)[%2]
      %inserted = tensor.insert %9 into %arg5[%10] : tensor<?xf64>
      %11 = polygeist.submap(%0, %arg4, %2, %3) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %12 = polygeist.submap(%inserted, %arg4, %2, %3) {map = #map2} : (tensor<?xf64>, index, index, index) -> tensor<?xf64>
      %13 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%11 : tensor<?xf64>) outs(%12 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %15 = arith.mulf %9, %in : f64
        %16 = arith.subf %out, %15 : f64
        %17 = linalg.index 0 : index
        %18 = affine.apply #map(%arg4)[%2]
        %19 = arith.cmpi slt, %17, %18 : index
        %20 = arith.select %19, %16, %out : f64
        linalg.yield %20 : f64
      } -> tensor<?xf64>
      %14 = polygeist.submapInverse(%inserted, %13, %arg4, %2, %3) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index, index, index) -> tensor<?xf64>
      affine.yield %14 : tensor<?xf64>
    }
    %5 = bufferization.to_memref %4 : memref<?xf64>
    memref.copy %5, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}

