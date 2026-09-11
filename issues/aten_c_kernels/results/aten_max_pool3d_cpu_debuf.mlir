#map = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 36 + s1 * 12 + s2 * 4)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_max_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2:2 = affine.for %arg3 = 0 to 2 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?xf32>, tensor<?xi32>) {
      %5:2 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?xf32>, tensor<?xi32>) {
        %6:2 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (tensor<?xf32>, tensor<?xi32>) {
          %alloca = memref.alloca(%c4) : memref<?xi32>
          %7 = bufferization.to_tensor %alloca : memref<?xi32>
          %alloca_0 = memref.alloca(%c4) : memref<?xf32>
          %8 = bufferization.to_tensor %alloca_0 : memref<?xf32>
          %9 = polygeist.submap(%arg10, %arg3, %arg6, %arg9, %c4) {map = #map} : (tensor<?xf32>, index, index, index, index) -> tensor<?xf32>
          %10 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%8 : tensor<?xf32>) outs(%9 : tensor<?xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<?xf32>
          %11 = polygeist.submapInverse(%arg10, %10, %arg3, %arg6, %arg9, %c4) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index, index, index, index) -> tensor<?xf32>
          %12 = polygeist.submap(%arg11, %arg3, %arg6, %arg9, %c4) {map = #map} : (tensor<?xi32>, index, index, index, index) -> tensor<?xi32>
          %13 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%7 : tensor<?xi32>) outs(%12 : tensor<?xi32>) {
          ^bb0(%in: i32, %out: i32):
            linalg.yield %in : i32
          } -> tensor<?xi32>
          %14 = polygeist.submapInverse(%arg11, %13, %arg3, %arg6, %arg9, %c4) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index, index, index, index) -> tensor<?xi32>
          affine.yield %11, %14 : tensor<?xf32>, tensor<?xi32>
        }
        affine.yield %6#0, %6#1 : tensor<?xf32>, tensor<?xi32>
      }
      affine.yield %5#0, %5#1 : tensor<?xf32>, tensor<?xi32>
    }
    %3 = bufferization.to_memref %2#1 : memref<?xi32>
    memref.copy %3, %arg2 : memref<?xi32> to memref<?xi32>
    %4 = bufferization.to_memref %2#0 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

