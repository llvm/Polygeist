#map = affine_map<(d0)[s0, s1] -> (d0 + s0 * 9 + s1 * 3)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %0) -> (tensor<?xf32>) {
      %3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %alloca = memref.alloca(%c3) : memref<?xi32>
        %4 = bufferization.to_tensor %alloca : memref<?xi32>
        %alloca_0 = memref.alloca(%c3) : memref<?xf32>
        %5 = bufferization.to_tensor %alloca_0 : memref<?xf32>
        %6 = polygeist.submap(%arg5, %arg2, %arg4, %c3) {map = #map} : (tensor<?xf32>, index, index, index) -> tensor<?xf32>
        %7 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%4, %5 : tensor<?xi32>, tensor<?xf32>) outs(%6 : tensor<?xf32>) {
        ^bb0(%in: i32, %in_1: f32, %out: f32):
          %9 = arith.sitofp %in : i32 to f32
          %10 = arith.divf %in_1, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<?xf32>
        %8 = polygeist.submapInverse(%arg5, %7, %arg2, %arg4, %c3) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index, index, index) -> tensor<?xf32>
        affine.yield %8 : tensor<?xf32>
      }
      affine.yield %3 : tensor<?xf32>
    }
    %2 = bufferization.to_memref %1 : memref<?xf32>
    memref.copy %2, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

