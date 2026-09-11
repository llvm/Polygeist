#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_standard_gamma_grad_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e-03 : f32
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%1, %0 : tensor<?xf32>, tensor<?xf32>) outs(%2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.subf %in, %in_0 : f32
      %6 = arith.addf %in_0, %cst : f32
      %7 = arith.divf %5, %6 : f32
      %8 = arith.addf %in, %cst : f32
      %9 = math.log %8 : f32
      %10 = arith.addf %7, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

