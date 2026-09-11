module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg7 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg8 restrict : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg9 restrict : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg10 restrict : memref<?x?xf64>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg0 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg6, %c0 : memref<?x?xf64>
    %9 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg6, %c1 : memref<?x?xf64>
    %10 = arith.index_cast %dim_0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg6 : memref<?x?xf64> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %13 = arith.muli %12, %c8_i64 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%9, %10, %15, %10) : (i32, i32, !llvm.ptr, i32) -> ()
    %16 = bufferization.to_tensor %arg6 restrict writable : memref<?x?xf64>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%8, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %2[0, 0] [%5, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %16[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_3 = arith.constant 1.000000e+00 : f64
    %subview = memref.subview %arg7[0, 0] [%8, %5] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_4 = memref.subview %arg8[0, 0] [%5, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %arg6[0, 0] [%8, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_6 = arith.constant 0 : index
    %dim_7 = memref.dim %subview, %c0_6 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %17 = arith.index_cast %dim_7 : index to i32
    %c1_8 = arith.constant 1 : index
    %dim_9 = memref.dim %subview, %c1_8 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %18 = arith.index_cast %dim_9 : index to i32
    %c1_10 = arith.constant 1 : index
    %dim_11 = memref.dim %subview_4, %c1_10 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %19 = arith.index_cast %dim_11 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_12 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %20 = arith.index_cast %intptr_12 : index to i64
    %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %21 = arith.index_cast %offset_14 : index to i64
    %c8_i64_17 = arith.constant 8 : i64
    %22 = arith.muli %21, %c8_i64_17 : i64
    %23 = arith.addi %20, %22 : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %25 = arith.index_cast %intptr_18 : index to i64
    %base_buffer_19, %offset_20, %sizes_21:2, %strides_22:2 = memref.extract_strided_metadata %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %26 = arith.index_cast %offset_20 : index to i64
    %c8_i64_23 = arith.constant 8 : i64
    %27 = arith.muli %26, %c8_i64_23 : i64
    %28 = arith.addi %25, %27 : i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %intptr_24 = memref.extract_aligned_pointer_as_index %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %30 = arith.index_cast %intptr_24 : index to i64
    %base_buffer_25, %offset_26, %sizes_27:2, %strides_28:2 = memref.extract_strided_metadata %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %31 = arith.index_cast %offset_26 : index to i64
    %c8_i64_29 = arith.constant 8 : i64
    %32 = arith.muli %31, %c8_i64_29 : i64
    %33 = arith.addi %30, %32 : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%17, %19, %18, %arg4, %24, %18, %29, %19, %cst_3, %34, %19) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %35 = bufferization.to_tensor %subview_5 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice = tensor.insert_slice %35 into %16[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %36 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %36, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    %extracted_slice_30 = tensor.extract_slice %3[0, 0] [%7, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_31 = tensor.extract_slice %4[0, 0] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_32 = arith.constant 1.000000e+00 : f64
    %subview_33 = memref.subview %arg9[0, 0] [%7, %6] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_34 = memref.subview %arg10[0, 0] [%8, %6] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_35 = arith.constant 0 : index
    %dim_36 = memref.dim %subview_5, %c0_35 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %37 = arith.index_cast %dim_36 : index to i32
    %c1_37 = arith.constant 1 : index
    %dim_38 = memref.dim %subview_5, %c1_37 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %38 = arith.index_cast %dim_38 : index to i32
    %c1_39 = arith.constant 1 : index
    %dim_40 = memref.dim %subview_33, %c1_39 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %39 = arith.index_cast %dim_40 : index to i32
    %intptr_41 = memref.extract_aligned_pointer_as_index %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %40 = arith.index_cast %intptr_41 : index to i64
    %base_buffer_42, %offset_43, %sizes_44:2, %strides_45:2 = memref.extract_strided_metadata %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %41 = arith.index_cast %offset_43 : index to i64
    %c8_i64_46 = arith.constant 8 : i64
    %42 = arith.muli %41, %c8_i64_46 : i64
    %43 = arith.addi %40, %42 : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
    %intptr_47 = memref.extract_aligned_pointer_as_index %subview_33 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %45 = arith.index_cast %intptr_47 : index to i64
    %base_buffer_48, %offset_49, %sizes_50:2, %strides_51:2 = memref.extract_strided_metadata %subview_33 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %46 = arith.index_cast %offset_49 : index to i64
    %c8_i64_52 = arith.constant 8 : i64
    %47 = arith.muli %46, %c8_i64_52 : i64
    %48 = arith.addi %45, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %intptr_53 = memref.extract_aligned_pointer_as_index %subview_34 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %50 = arith.index_cast %intptr_53 : index to i64
    %base_buffer_54, %offset_55, %sizes_56:2, %strides_57:2 = memref.extract_strided_metadata %subview_34 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %51 = arith.index_cast %offset_55 : index to i64
    %c8_i64_58 = arith.constant 8 : i64
    %52 = arith.muli %51, %c8_i64_58 : i64
    %53 = arith.addi %50, %52 : i64
    %54 = llvm.inttoptr %53 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%37, %39, %38, %cst_32, %44, %38, %49, %39, %arg5, %54, %39) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %55 = bufferization.to_tensor %subview_34 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %56 = bufferization.to_tensor %arg10 restrict writable : memref<?x?xf64>
    %57 = bufferization.to_memref %56 : memref<?x?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %57, %arg10 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

