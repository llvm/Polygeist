module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %6:2 = affine.for %arg6 = 0 to %5 iter_args(%arg7 = %2, %arg8 = %0) -> (tensor<?xf64>, tensor<?x?x?xf64>) {
      %9:2 = affine.for %arg9 = 0 to %3 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (tensor<?xf64>, tensor<?x?x?xf64>) {
        %extracted_slice = tensor.extract_slice %arg11[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?xf64>
        %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%4, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
        %extracted_slice_1 = tensor.extract_slice %arg10[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
        %inserted_slice = tensor.insert_slice %extracted_slice_1 into %arg10[0] [%4] [1] : tensor<?xf64> into tensor<?xf64>
        %extracted_slice_2 = tensor.extract_slice %arg11[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?xf64>
        %cst_3 = arith.constant 1.000000e+00 : f64
        %cst_4 = arith.constant 0.000000e+00 : f64
        %10 = arith.index_cast %4 : index to i32
        %11 = arith.index_cast %4 : index to i32
        %c1 = arith.constant 1 : index
        %dim = tensor.dim %1, %c1 : tensor<?x?xf64>
        %12 = arith.index_cast %dim : index to i32
        %intptr = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
        %13 = arith.index_cast %intptr : index to i64
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
        %14 = arith.index_cast %offset : index to i64
        %c0_i64 = arith.constant 0 : i64
        %15 = arith.index_cast %strides#0 : index to i64
        %16 = arith.muli %c0_i64, %15 : i64
        %17 = arith.addi %14, %16 : i64
        %c0_i64_5 = arith.constant 0 : i64
        %18 = arith.index_cast %strides#1 : index to i64
        %19 = arith.muli %c0_i64_5, %18 : i64
        %20 = arith.addi %17, %19 : i64
        %c8_i64 = arith.constant 8 : i64
        %21 = arith.muli %20, %c8_i64 : i64
        %22 = arith.addi %13, %21 : i64
        %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
        %intptr_6 = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?x?xf64> -> index
        %24 = arith.index_cast %intptr_6 : index to i64
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg3 : memref<?x?x?xf64> -> memref<f64>, index, index, index, index, index, index, index
        %25 = arith.index_cast %offset_8 : index to i64
        %26 = arith.index_cast %arg6 : index to i64
        %27 = arith.index_cast %strides_10#0 : index to i64
        %28 = arith.muli %26, %27 : i64
        %29 = arith.addi %25, %28 : i64
        %30 = arith.index_cast %arg9 : index to i64
        %31 = arith.index_cast %strides_10#1 : index to i64
        %32 = arith.muli %30, %31 : i64
        %33 = arith.addi %29, %32 : i64
        %c0_i64_11 = arith.constant 0 : i64
        %34 = arith.index_cast %strides_10#2 : index to i64
        %35 = arith.muli %c0_i64_11, %34 : i64
        %36 = arith.addi %33, %35 : i64
        %c8_i64_12 = arith.constant 8 : i64
        %37 = arith.muli %36, %c8_i64_12 : i64
        %38 = arith.addi %24, %37 : i64
        %39 = llvm.inttoptr %38 : i64 to !llvm.ptr
        %subview = memref.subview %arg3[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : memref<?x?x?xf64> to memref<?xf64, strided<[1], offset: ?>>
        %intptr_13 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1], offset: ?>> -> index
        %40 = arith.index_cast %intptr_13 : index to i64
        %base_buffer_14, %offset_15, %sizes_16, %strides_17 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1], offset: ?>> -> memref<f64>, index, index, index
        %41 = arith.index_cast %offset_15 : index to i64
        %c8_i64_18 = arith.constant 8 : i64
        %42 = arith.muli %41, %c8_i64_18 : i64
        %43 = arith.addi %40, %42 : i64
        %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
        func.call @polygeist_cublas_dgemv_T(%10, %11, %cst_3, %23, %12, %39, %cst_4, %44) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
        %inserted_slice_19 = tensor.insert_slice %extracted_slice_2 into %arg11[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : tensor<?xf64> into tensor<?x?x?xf64>
        affine.yield %arg10, %arg11 : tensor<?xf64>, tensor<?x?x?xf64>
      }
      affine.yield %9#0, %9#1 : tensor<?xf64>, tensor<?x?x?xf64>
    }
    %7 = bufferization.to_memref %6#1 : memref<?x?x?xf64>
    memref.copy %7, %arg3 : memref<?x?x?xf64> to memref<?x?x?xf64>
    %8 = bufferization.to_memref %6#0 : memref<?xf64>
    memref.copy %8, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
}

