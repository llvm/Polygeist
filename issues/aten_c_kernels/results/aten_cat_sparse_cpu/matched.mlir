#map = affine_map<(d0, d1) -> (d1 + d0 * 256)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cat_sparse_cpu(%arg0: memref<?x256xi32>, %arg1: memref<?x256xf32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x256xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?x256xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xi32>
    %3 = bufferization.to_tensor %arg3 : memref<?xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c4, %c256] [1, 1] : tensor<?x256xi32> to tensor<?x?xi32>
    %4 = polygeist.submap(%2, %c4, %c256) {map = #map} : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice : tensor<?x?xi32>) outs(%4 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?x?xi32>
    %6 = polygeist.submapInverse(%2, %5, %c4, %c256) {map = #map} : (tensor<?xi32>, tensor<?x?xi32>, index, index) -> tensor<?xi32>
    %7 = bufferization.to_memref %6 : memref<?xi32>
    memref.copy %7, %arg2 : memref<?xi32> to memref<?xi32>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%c4, %c256] [1, 1] : tensor<?x256xf32> to tensor<?x?xf32>
    %8 = polygeist.submap(%3, %c4, %c256) {map = #map} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
    %9 = kernel.launch @cudaCopy2D_f32_tensor(%extracted_slice_0, %8) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %10 = polygeist.submapInverse(%3, %9, %c4, %c256) {map = #map} : (tensor<?xf32>, tensor<?x?xf32>, index, index) -> tensor<?xf32>
    %11 = bufferization.to_memref %10 : memref<?xf32>
    memref.copy %11, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

