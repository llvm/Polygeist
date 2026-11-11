#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cub_reduce_value_sum_f32(%arg0: i32, %arg1: f32, %arg2: memref<?xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %arg1 into %2[] : tensor<f32>
    %3 = polygeist.submap(%0, %1) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%3 : tensor<?xf32>) outs(%inserted : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %out, %in : f32
      linalg.yield %5 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %4[] : tensor<f32>
    return %extracted : f32
  }
}

