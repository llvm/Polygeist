module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = tensor.empty() : tensor<f64>
    %5 = llvm.mlir.undef : f64
    %inserted = tensor.insert %5 into %4[] : tensor<f64>
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.subi %6, %c1 : index
    %8 = arith.subi %6, %c1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg4, %c0 : memref<?x?xf64>
    %9 = arith.index_cast %dim : index to i32
    %c1_0 = arith.constant 1 : index
    %dim_1 = memref.dim %arg4, %c1_0 : memref<?x?xf64>
    %10 = arith.index_cast %dim_1 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %13 = arith.muli %12, %c8_i64 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %16 = arith.index_cast %strides_5#0 : index to i32
    %intptr_6 = memref.extract_aligned_pointer_as_index %arg6 : memref<?x?xf64> -> index
    %17 = arith.index_cast %intptr_6 : index to i64
    %base_buffer_7, %offset_8, %sizes_9:2, %strides_10:2 = memref.extract_strided_metadata %arg6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %18 = arith.index_cast %offset_8 : index to i64
    %c8_i64_11 = arith.constant 8 : i64
    %19 = arith.muli %18, %c8_i64_11 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %arg6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %22 = arith.index_cast %strides_15#0 : index to i32
    %intptr_16 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %23 = arith.index_cast %intptr_16 : index to i64
    %base_buffer_17, %offset_18, %sizes_19:2, %strides_20:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %24 = arith.index_cast %offset_18 : index to i64
    %c8_i64_21 = arith.constant 8 : i64
    %25 = arith.muli %24, %c8_i64_21 : i64
    %26 = arith.addi %23, %25 : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    %base_buffer_22, %offset_23, %sizes_24:2, %strides_25:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %28 = arith.index_cast %strides_25#0 : index to i32
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dsymm_left_lower_row_major(%9, %10, %arg2, %15, %16, %21, %22, %arg3, %27, %28) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dsymm_left_lower_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

