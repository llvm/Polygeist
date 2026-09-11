module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_min_values_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c63 = arith.constant 63 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c32, 1] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0] [%c32] [1] : tensor<?xf32> to tensor<?xf32>
    %cast = memref.cast %arg0 : memref<?x64xf32> to memref<?x?xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %dim = memref.dim %cast, %c0 : memref<?x?xf32>
    %2 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_1 = memref.dim %cast, %c1 : memref<?x?xf32>
    %3 = arith.index_cast %dim_1 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %cast : memref<?x?xf32> -> index
    %4 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %cast : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %5 = arith.index_cast %offset : index to i64
    %c4_i64 = arith.constant 4 : i64
    %6 = arith.muli %5, %c4_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg1 : memref<?xf32> -> index
    %9 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg1 : memref<?xf32> -> memref<f32>, index, index, index
    %10 = arith.index_cast %offset_4 : index to i64
    %c4_i64_7 = arith.constant 4 : i64
    %11 = arith.muli %10, %c4_i64_7 : i64
    %12 = arith.addi %9, %11 : i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cub_segmented_reduce_f32(%c1_i32, %2, %3, %8, %13) : (i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cub_segmented_reduce_f32(i32, i32, i32, !llvm.ptr, !llvm.ptr)
}
