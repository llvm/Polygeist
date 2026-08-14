#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4 + d2 * 2, d5 + d3 * 2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool2d(%arg0: memref<?x4x16x16xf32>, %arg1: memref<?x4x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x4x16x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x4x8x8xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0, 0, 0] [%c2, %c4, %c8, %c8] [1, 1, 1, 1] : tensor<?x4x8x8xf32> to tensor<?x?x?x?xf32>
    %3 = polygeist.submap(%0, %c2, %c4, %c8, %c8, %c2, %c2) {map = #map1} : (tensor<?x4x16x16xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %v0_tc0 = tensor.cast %0 : tensor<?x4x16x16xf32> to tensor<?x?x?x?xf32>

    %winconv0_weight = arith.constant 0.25 : f32

    %winconv0_kh = arith.constant 2 : i32

    %winconv0_kw = arith.constant 2 : i32

    %winconv0_sh = arith.constant 2 : i32

    %winconv0_sw = arith.constant 2 : i32

    %winconv0_dh = arith.constant 1 : i32

    %winconv0_dw = arith.constant 1 : i32

    %winconv0_ph = arith.constant 0 : i32

    %winconv0_pw = arith.constant 0 : i32

    %4 = kernel.launch @cudnnConvolution2DWindow_f32(%v0_tc0, %extracted_slice, %winconv0_weight, %winconv0_kh, %winconv0_kw, %winconv0_sh, %winconv0_sw, %winconv0_dh, %winconv0_dw, %winconv0_ph, %winconv0_pw) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32, i32, i32, i32, i32, i32, i32, i32, i32) -> tensor<?x?x?x?xf32>
    %inserted_slice = tensor.insert_slice %4 into %1[0, 0, 0, 0] [%c2, %c4, %c8, %c8] [1, 1, 1, 1] : tensor<?x?x?x?xf32> into tensor<?x4x8x8xf32>
    %5 = bufferization.to_memref %inserted_slice : memref<?x4x8x8xf32>
    memref.copy %5, %arg1 : memref<?x4x8x8xf32> to memref<?x4x8x8xf32>
    return
  }
}

