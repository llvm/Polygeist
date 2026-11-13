module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d(%arg0: memref<?x3x4x4xf32>, %arg1: memref<?x3x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x3x8x8xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [%c2, %c3, %c8, %c8] [1, 1, 1, 1] : tensor<?x3x8x8xf32> to tensor<?x?x?x?xf32>
    %c6_i32 = arith.constant 6 : i32
    %c2_i32_1 = arith.constant 2 : i32
    %c6_i32_2 = arith.constant 6 : i32
    %c1_i32_3 = arith.constant 1 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c4_i32_5 = arith.constant 4 : i32
    %c1_i32_6 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8_i32_7 = arith.constant 8 : i32
    %c1_i32_8 = arith.constant 1 : i32
    %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<?x3x4x4xf32> -> index
    %1 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %arg0 : memref<?x3x4x4xf32> -> memref<f32>, index, index, index, index, index, index, index, index, index
    %2 = arith.index_cast %offset : index to i64
    %c4_i64 = arith.constant 4 : i64
    %3 = arith.muli %2, %c4_i64 : i64
    %4 = arith.addi %1, %3 : i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %arg1 : memref<?x3x8x8xf32> -> index
    %6 = arith.index_cast %intptr_9 : index to i64
    %base_buffer_10, %offset_11, %sizes_12:4, %strides_13:4 = memref.extract_strided_metadata %arg1 : memref<?x3x8x8xf32> -> memref<f32>, index, index, index, index, index, index, index, index, index
    %7 = arith.index_cast %offset_11 : index to i64
    %c4_i64_14 = arith.constant 4 : i64
    %8 = arith.muli %7, %c4_i64_14 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %11 = llvm.mlir.zero : !llvm.ptr
    call @polygeist_cudnn_adaptive_pool_f32(%c6_i32, %c2_i32_1, %c6_i32_2, %c1_i32_3, %c4_i32_4, %c4_i32_5, %c1_i32_6, %c8_i32, %c8_i32_7, %c1_i32_8, %5, %10, %11) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cudnn_adaptive_pool_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
}
