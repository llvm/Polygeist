#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1 + d0 * 16, d2 + d0 * 16)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_block_diag_cpu(%arg0: memref<?x16x16xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x16x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%c64, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %2 = kernel.launch @memset_zero_2D_f32(%extracted_slice) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %2 into %1[0, 0] [%c64, %c64] [1, 1] : tensor<?x?xf32> into tensor<?x64xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, 0, 0] [%c4, %c16, %c16] [1, 1, 1] : tensor<?x16x16xf32> to tensor<?x?x?xf32>
    %3 = polygeist.submap(%inserted_slice, %c4, %c16, %c16) {map = #map1} : (tensor<?x64xf32>, index, index, index) -> tensor<?x?x?xf32>
    %4 = kernel.launch @cutensorPermute_f32_r3_tensor(%extracted_slice_0, %3) {cutensor_input_modes = array<i64: 0, 1, 2>, cutensor_output_modes = array<i64: 0, 1, 2>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %5 = polygeist.submapInverse(%inserted_slice, %4, %c4, %c16, %c16) {map = #map1} : (tensor<?x64xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x64xf32>
    %6 = bufferization.to_memref %5 : memref<?x64xf32>
    memref.copy %6, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

