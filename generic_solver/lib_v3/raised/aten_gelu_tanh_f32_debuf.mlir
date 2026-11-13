#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_gelu_tanh_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 4.471500e-02 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0.79788456080286541 : f64
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.truncf %cst_2 : f64 to f32
    %4 = polygeist.submap(%1, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %5 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%4 : tensor<?xf32>) outs(%5 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.mulf %in, %in : f32
      %10 = arith.mulf %9, %in : f32
      %11 = arith.mulf %10, %cst : f32
      %12 = arith.addf %in, %11 : f32
      %13 = arith.mulf %3, %12 : f32
      %14 = arith.mulf %in, %cst_0 : f32
      %15 = math.tanh %13 : f32
      %16 = arith.addf %15, %cst_1 : f32
      %17 = arith.mulf %14, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<?xf32>
    %7 = polygeist.submapInverse(%0, %6, %2) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %8 = bufferization.to_memref %7 : memref<?xf32>
    memref.copy %8, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

