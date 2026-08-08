#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1)[s0] -> (s0, d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_embedding_bag_per_sample_backward_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?x16xi32>, %arg3: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x16xf32>
    %1 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %0) -> (tensor<?x16xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, %c16] [1, 1] : tensor<?x16xf32> to tensor<?xf32>
      %3 = kernel.launch @memset_zero_1D_f32(%extracted_slice) : (tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg5[%arg4, 0] [1, %c16] [1, 1] : tensor<?xf32> into tensor<?x16xf32>
      %subview = memref.subview %arg2[%arg4, 0] [1, %c16] [1, 1] : memref<?x16xi32> to memref<?xi32, strided<[1], offset: ?>>
      %cast = memref.cast %subview : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32>
      %4 = polygeist.submap(%cast, %c16, %c64) {map = #map1} : (memref<?xi32>, index, index) -> memref<?x?xi32>
      %5 = polygeist.submap(%inserted_slice, %arg4, %c16, %c64) {map = #map2} : (tensor<?x16xf32>, index, index, index) -> tensor<?x?xf32>
      %6 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%4 : memref<?x?xi32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: i32, %out: f32):
        %8 = arith.index_cast %in : i32 to index
        %9 = linalg.index 1 : index
        %10 = memref.load %arg0[%arg4, %9] : memref<?x64xf32>
        %11 = memref.load %arg1[%8, %9] : memref<?x64xf32>
        %12 = arith.mulf %10, %11 : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      } -> tensor<?x?xf32>
      %7 = polygeist.submapInverse(%inserted_slice, %6, %arg4, %c16, %c64) {map = #map2} : (tensor<?x16xf32>, tensor<?x?xf32>, index, index, index) -> tensor<?x16xf32>
      affine.yield %7 : tensor<?x16xf32>
    }
    %2 = bufferization.to_memref %1 : memref<?x16xf32>
    memref.copy %2, %arg3 : memref<?x16xf32> to memref<?x16xf32>
    return
  }
}

