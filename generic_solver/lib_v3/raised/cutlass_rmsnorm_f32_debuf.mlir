#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cutlass_rmsnorm_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %cst into %4[] : tensor<f32>
    %5 = polygeist.submap(%2, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%5 : tensor<?xf32>) outs(%inserted : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.mulf %in, %in : f32
      %18 = arith.addf %out, %17 : f32
      linalg.yield %18 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %6[] : tensor<f32>
    %7 = arith.sitofp %arg0 : i32 to f32
    %8 = arith.divf %extracted, %7 : f32
    %9 = arith.addf %8, %cst_0 : f32
    %10 = math.sqrt %9 : f32
    %11 = polygeist.submap(%2, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %12 = polygeist.submap(%1, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %13 = polygeist.submap(%0, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %14 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%11, %12 : tensor<?xf32>, tensor<?xf32>) outs(%13 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %17 = arith.divf %in, %10 : f32
      %18 = arith.mulf %17, %in_1 : f32
      linalg.yield %18 : f32
    } -> tensor<?xf32>
    %15 = polygeist.submapInverse(%0, %14, %3) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %16 = bufferization.to_memref %15 : memref<?xf32>
    memref.copy %16, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

