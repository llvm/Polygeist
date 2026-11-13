#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_addr_elementwise_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: f32, %arg5: f32, %arg6: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg6 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xf32>
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.cmpf oeq, %arg4, %cst : f32
    %6 = polygeist.submap(%2, %4) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %7 = polygeist.submap(%1, %4) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %8 = polygeist.submap(%3, %4) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %9 = polygeist.submap(%0, %4) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %10 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%6, %7, %8 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%9 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %13 = arith.mulf %arg5, %in : f32
      %14 = arith.mulf %13, %in_0 : f32
      %15 = arith.mulf %arg4, %in_1 : f32
      %16 = arith.mulf %arg5, %in : f32
      %17 = arith.mulf %16, %in_0 : f32
      %18 = arith.addf %15, %17 : f32
      %19 = arith.select %5, %14, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<?xf32>
    %11 = polygeist.submapInverse(%0, %10, %4) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %12 = bufferization.to_memref %11 : memref<?xf32>
    memref.copy %12, %arg6 : memref<?xf32> to memref<?xf32>
    return
  }
}

