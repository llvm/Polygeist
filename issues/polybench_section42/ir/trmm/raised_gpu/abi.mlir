module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg4, %c0 : memref<?x?xf64>
    %4 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg4, %c1 : memref<?x?xf64>
    %5 = arith.index_cast %dim_0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %7 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %8 = arith.muli %7, %c8_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %11 = arith.index_cast %strides_4#0 : index to i32
    %intptr_5 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %12 = arith.index_cast %intptr_5 : index to i64
    %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %13 = arith.index_cast %offset_7 : index to i64
    %c8_i64_10 = arith.constant 8 : i64
    %14 = arith.muli %13, %c8_i64_10 : i64
    %15 = arith.addi %12, %14 : i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %17 = arith.index_cast %strides_14#0 : index to i32
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dtrmm_left_lower_trans_unit_row_major(%4, %5, %arg2, %10, %11, %16, %17) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dtrmm_left_lower_trans_unit_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

