#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_hardswish_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %cst_1 = arith.constant 6.000000e+00 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = polygeist.submap(%1, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %4 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%3 : tensor<?xf32>) outs(%4 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.addf %in, %cst_0 : f32
      %9 = arith.cmpf olt, %8, %cst : f32
      %10 = arith.select %9, %cst, %8 : f32
      %11 = arith.cmpf olt, %cst_1, %10 : f32
      %12 = arith.select %11, %cst_1, %10 : f32
      %13 = arith.mulf %in, %12 : f32
      %14 = arith.divf %13, %cst_1 : f32
      linalg.yield %14 : f32
    } -> tensor<?xf32>
    %6 = polygeist.submapInverse(%0, %5, %2) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %7 = bufferization.to_memref %6 : memref<?xf32>
    memref.copy %7, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

