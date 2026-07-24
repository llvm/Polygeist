#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x8x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg5 : memref<?x8x16x16xf32>
    %1 = bufferization.to_tensor %arg4 : memref<?xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?xf32>
    %3 = bufferization.to_tensor %arg2 : memref<?xf32>
    %4 = bufferization.to_tensor %arg1 : memref<?xf32>
    %5 = bufferization.to_tensor %arg0 : memref<?x8x16x16xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%4, %5, %3, %2, %1 : tensor<?xf32>, tensor<?x8x16x16xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?x8x16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %in_3: f32, %out: f32):
      %8 = arith.subf %in_0, %in_1 : f32
      %9 = arith.mulf %in, %8 : f32
      %10 = arith.mulf %9, %in_2 : f32
      %11 = arith.addf %10, %in_3 : f32
      linalg.yield %11 : f32
    } -> tensor<?x8x16x16xf32>
    %7 = bufferization.to_memref %6 : memref<?x8x16x16xf32>
    memref.copy %7, %arg5 : memref<?x8x16x16xf32> to memref<?x8x16x16xf32>
    return
  }
}

