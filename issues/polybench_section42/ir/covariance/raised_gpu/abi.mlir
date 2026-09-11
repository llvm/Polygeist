module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %0 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %1 = arith.index_cast %dim_0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?xf64> -> index
    %2 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %3 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %4 = arith.muli %3, %c8_i64 : i64
    %5 = arith.addi %2, %4 : i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %7 = arith.index_cast %strides_4#0 : index to i32
    %intptr_5 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %8 = arith.index_cast %intptr_5 : index to i64
    %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %9 = arith.index_cast %offset_7 : index to i64
    %c8_i64_10 = arith.constant 8 : i64
    %10 = arith.muli %9, %c8_i64_10 : i64
    %11 = arith.addi %8, %10 : i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %13 = arith.index_cast %strides_14#0 : index to i32
    %intptr_15 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %14 = arith.index_cast %intptr_15 : index to i64
    %base_buffer_16, %offset_17, %sizes_18, %strides_19 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %15 = arith.index_cast %offset_17 : index to i64
    %c8_i64_20 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64_20 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dcovariance_row_major(%0, %1, %arg2, %6, %7, %12, %13, %18) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dcovariance_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

