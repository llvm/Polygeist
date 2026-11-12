#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_shrink_backward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.negf %arg4 : f32
    %5 = polygeist.submap(%1, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %6 = polygeist.submap(%2, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %7 = polygeist.submap(%0, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %8 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%5, %6 : tensor<?xf32>, tensor<?xf32>) outs(%7 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.cmpf oge, %in, %4 : f32
      %12 = arith.cmpf ole, %in, %arg4 : f32
      %13 = arith.select %11, %12, %false : i1
      %14 = arith.select %13, %cst, %in_0 : f32
      linalg.yield %14 : f32
    } -> tensor<?xf32>
    %9 = polygeist.submapInverse(%0, %8, %3) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %10 = bufferization.to_memref %9 : memref<?xf32>
    memref.copy %10, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

