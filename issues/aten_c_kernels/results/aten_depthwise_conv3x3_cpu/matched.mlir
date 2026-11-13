#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (0, d0, d3 + d1 - 1, d4 + d2 - 1)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (-d0 - d1 + 16)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0 + d1 - 1)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 + d3 - 1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (-d2 - d3 + 16)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_depthwise_conv3x3_cpu(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x8x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x8x16x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x3x3xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?x8x16x16xf32>
    %extracted_slice = tensor.extract_slice %2[0] [%c8] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %3[0, 0, 0, 0] [1, %c8, %c16, %c16] [1, 1, 1, 1] : tensor<?x8x16x16xf32> to tensor<?x?x?xf32>
    %fixed_conv_6_0 = memref.cast %arg0 : memref<?x8x16x16xf32> to memref<?x?x?x?xf32>

    %fixed_conv_6_1 = memref.cast %arg1 : memref<?x3x3xf32> to memref<?x?x?xf32>

    %fixed_conv_6_2 = memref.cast %arg2 : memref<?xf32> to memref<?xf32>

    %fixed_conv_6_3 = memref.cast %arg3 : memref<?x8x16x16xf32> to memref<?x?x?x?xf32>

    kernel.launch @cudnnDepthwiseConvolution2D_f32_memref(%fixed_conv_6_0, %fixed_conv_6_1, %fixed_conv_6_2, %fixed_conv_6_3) : (memref<?x?x?x?xf32>, memref<?x?x?xf32>, memref<?xf32>, memref<?x?x?x?xf32>) -> ()
    return
  }
}

