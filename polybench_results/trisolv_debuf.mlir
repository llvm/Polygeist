#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x40xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg1 : memref<?x40xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %1) -> (tensor<?xf64>) {
      %extracted = tensor.extract %0[%arg4] : tensor<?xf64>
      %inserted = tensor.insert %extracted into %arg5[%arg4] : tensor<?xf64>
      %6 = arith.subi %3, %c1 : index
      %7 = polygeist.submap(%2, %arg4, %6) {map = #map} : (tensor<?x40xf64>, index, index) -> tensor<?xf64>
      %8 = polygeist.submap(%inserted, %6) {map = #map1} : (tensor<?xf64>, index) -> tensor<?xf64>
      %9 = polygeist.submap(%inserted, %arg4, %6) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %10 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%7, %8 : tensor<?xf64>, tensor<?xf64>) outs(%9 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_3: f64, %out: f64):
        %13 = arith.mulf %in, %in_3 : f64
        %14 = arith.subf %out, %13 : f64
        %15 = linalg.index 0 : index
        %16 = arith.cmpi slt, %15, %arg4 : index
        %17 = arith.select %16, %14, %out : f64
        linalg.yield %17 : f64
      } -> tensor<?xf64>
      %11 = polygeist.submapInverse(%inserted, %10, %arg4, %6) {map = #map2} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
      %extracted_0 = tensor.extract %11[%arg4] : tensor<?xf64>
      %extracted_1 = tensor.extract %2[%arg4, %arg4] : tensor<?x40xf64>
      %12 = arith.divf %extracted_0, %extracted_1 : f64
      %inserted_2 = tensor.insert %12 into %11[%arg4] : tensor<?xf64>
      affine.yield %inserted_2 : tensor<?xf64>
    }
    %5 = bufferization.to_memref %4 : memref<?xf64>
    memref.copy %5, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

