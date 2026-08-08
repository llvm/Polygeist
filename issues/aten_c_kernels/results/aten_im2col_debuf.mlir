#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4 + d2, d5 + d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_im2col(%arg0: memref<?x3x8x8xf32>, %arg1: memref<?x3x3x3x6x6xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x3x8x8xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x3x3x3x6x6xf32>
    %2 = polygeist.submap(%0, %c2, %c3, %c3, %c3, %c6, %c6) {map = #map} : (tensor<?x3x8x8xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0, 0, 0, 0, 0] [%c2, %c3, %c3, %c3, %c6, %c6] [1, 1, 1, 1, 1, 1] : tensor<?x3x3x3x6x6xf32> to tensor<?x?x?x?x?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%2 : tensor<?x?x?x?x?x?xf32>) outs(%extracted_slice : tensor<?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?x?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0, 0, 0, 0] [%c2, %c3, %c3, %c3, %c6, %c6] [1, 1, 1, 1, 1, 1] : tensor<?x?x?x?x?x?xf32> into tensor<?x3x3x3x6x6xf32>
    %4 = bufferization.to_memref %inserted_slice : memref<?x3x3x3x6x6xf32>
    memref.copy %4, %arg1 : memref<?x3x3x3x6x6xf32> to memref<?x3x3x3x6x6xf32>
    return
  }
}

