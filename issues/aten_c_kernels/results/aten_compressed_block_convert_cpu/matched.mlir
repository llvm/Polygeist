module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_compressed_block_convert_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x16x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c-1 = arith.constant -1 : index

    %block_permute_1353_input_view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 4, 16, 4], strides: [256, 64, 4, 1] : memref<?x64xf32> to memref<16x4x16x4xf32, strided<[256, 64, 4, 1]>>
    %block_permute_1353_input_static = bufferization.to_tensor %block_permute_1353_input_view restrict : memref<16x4x16x4xf32, strided<[256, 64, 4, 1]>>
    %block_permute_1353_input = tensor.cast %block_permute_1353_input_static : tensor<16x4x16x4xf32> to tensor<?x?x?x?xf32>
    %block_permute_1353_output_static = bufferization.to_tensor %arg1 restrict writable : memref<?x16x4x4xf32>
    %block_permute_1353_output = tensor.cast %block_permute_1353_output_static : tensor<?x16x4x4xf32> to tensor<?x?x?x?xf32>
    %block_permute_1353_result = kernel.launch @cutensorPermute_f32_r4_tensor(%block_permute_1353_input, %block_permute_1353_output) {cutensor_input_modes = array<i64: 0, 2, 1, 3>, cutensor_output_modes = array<i64: 0, 1, 2, 3>} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %block_permute_1353_result_static = tensor.cast %block_permute_1353_result : tensor<?x?x?x?xf32> to tensor<?x16x4x4xf32>
    %block_permute_1353_result_memref = bufferization.to_memref %block_permute_1353_result_static : memref<?x16x4x4xf32>
    memref.copy %block_permute_1353_result_memref, %arg1 : memref<?x16x4x4xf32> to memref<?x16x4x4xf32>
    return
  }
}

