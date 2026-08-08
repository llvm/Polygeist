#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5 + d1, d6 + d2, d7 + d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d4, d5, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_slow_conv3d_forward_cpu(%arg0: memref<?x8x9x10xf32>, %arg1: memref<?x2x3x3x3xf32>, %arg2: memref<?x6x7x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x8x9x10xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x2x3x3x3xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x6x7x8xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0, 0, 0] [%c3, %c6, %c7, %c8] [1, 1, 1, 1] : tensor<?x6x7x8xf32> to tensor<?x?x?x?xf32>
    %4 = polygeist.submap(%0, %c3, %c6, %c7, %c8, %c2, %c3, %c3, %c3) {map = #map1} : (tensor<?x8x9x10xf32>, index, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0, 0, 0] [%c3, %c2, %c3, %c3, %c3] [1, 1, 1, 1, 1] : tensor<?x2x3x3x3xf32> to tensor<?x?x?x?x?xf32>
    %5 = kernel.launch @cudnnConvolution3D_f32(%4, %extracted_slice_0, %extracted_slice) : (tensor<?x?x?x?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %inserted_slice = tensor.insert_slice %5 into %2[0, 0, 0, 0] [%c3, %c6, %c7, %c8] [1, 1, 1, 1] : tensor<?x?x?x?xf32> into tensor<?x6x7x8xf32>
    %6 = bufferization.to_memref %inserted_slice : memref<?x6x7x8xf32>
    memref.copy %6, %arg2 : memref<?x6x7x8xf32> to memref<?x6x7x8xf32>
    return
  }
}

