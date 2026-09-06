module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg6 restrict : memref<?xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?xf64>
    %6 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %7 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
    %8 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = arith.addi %7, %9 : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%6, %11) : (i32, !llvm.ptr) -> ()
    %12 = bufferization.to_tensor %arg3 restrict writable : memref<?xf64>
    %13 = arith.index_cast %arg1 : i32 to index
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg4, %c0_0 : memref<?xf64>
    %14 = arith.index_cast %dim_1 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %15 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %16 = arith.index_cast %offset_4 : index to i64
    %c8_i64_7 = arith.constant 8 : i64
    %17 = arith.muli %16, %c8_i64_7 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%14, %19) : (i32, !llvm.ptr) -> ()
    %20 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %extracted_slice = tensor.extract_slice %4[0] [%13] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_8 = tensor.extract_slice %0[0, 0] [%13, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_9 = tensor.extract_slice %12[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_10 = arith.constant 1.000000e+00 : f64
    %21 = arith.index_cast %13 : index to i32
    %22 = arith.index_cast %5 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_11 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %23 = arith.index_cast %intptr_11 : index to i64
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %24 = arith.index_cast %offset_13 : index to i64
    %c0_i64 = arith.constant 0 : i64
    %25 = arith.index_cast %strides_15#0 : index to i64
    %26 = arith.muli %c0_i64, %25 : i64
    %27 = arith.addi %24, %26 : i64
    %c0_i64_16 = arith.constant 0 : i64
    %28 = arith.index_cast %strides_15#1 : index to i64
    %29 = arith.muli %c0_i64_16, %28 : i64
    %30 = arith.addi %27, %29 : i64
    %c8_i64_17 = arith.constant 8 : i64
    %31 = arith.muli %30, %c8_i64_17 : i64
    %32 = arith.addi %23, %31 : i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %34 = arith.index_cast %intptr_18 : index to i64
    %base_buffer_19, %offset_20, %sizes_21, %strides_22 = memref.extract_strided_metadata %arg6 : memref<?xf64> -> memref<f64>, index, index, index
    %35 = arith.index_cast %offset_20 : index to i64
    %c0_i64_23 = arith.constant 0 : i64
    %36 = arith.index_cast %strides_22 : index to i64
    %37 = arith.muli %c0_i64_23, %36 : i64
    %38 = arith.addi %35, %37 : i64
    %c8_i64_24 = arith.constant 8 : i64
    %39 = arith.muli %38, %c8_i64_24 : i64
    %40 = arith.addi %34, %39 : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %intptr_25 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %42 = arith.index_cast %intptr_25 : index to i64
    %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
    %43 = arith.index_cast %offset_27 : index to i64
    %c0_i64_30 = arith.constant 0 : i64
    %44 = arith.index_cast %strides_29 : index to i64
    %45 = arith.muli %c0_i64_30, %44 : i64
    %46 = arith.addi %43, %45 : i64
    %c8_i64_31 = arith.constant 8 : i64
    %47 = arith.muli %46, %c8_i64_31 : i64
    %48 = arith.addi %42, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%21, %22, %cst_10, %33, %22, %41, %cst_10, %49) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview = memref.subview %arg3[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %50 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %51 = bufferization.to_tensor %arg3 restrict writable : memref<?xf64>
    %52 = bufferization.to_memref %51 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %52, %arg3 : memref<?xf64> to memref<?xf64>
    %extracted_slice_32 = tensor.extract_slice %0[0, 0] [%13, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_34 = tensor.extract_slice %20[0] [%13] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_35 = arith.constant 1.000000e+00 : f64
    %53 = arith.index_cast %13 : index to i32
    %54 = arith.index_cast %5 : index to i32
    %intptr_36 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %55 = arith.index_cast %intptr_36 : index to i64
    %base_buffer_37, %offset_38, %sizes_39:2, %strides_40:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %56 = arith.index_cast %offset_38 : index to i64
    %c0_i64_41 = arith.constant 0 : i64
    %57 = arith.index_cast %strides_40#0 : index to i64
    %58 = arith.muli %c0_i64_41, %57 : i64
    %59 = arith.addi %56, %58 : i64
    %c0_i64_42 = arith.constant 0 : i64
    %60 = arith.index_cast %strides_40#1 : index to i64
    %61 = arith.muli %c0_i64_42, %60 : i64
    %62 = arith.addi %59, %61 : i64
    %c8_i64_43 = arith.constant 8 : i64
    %63 = arith.muli %62, %c8_i64_43 : i64
    %64 = arith.addi %55, %63 : i64
    %65 = llvm.inttoptr %64 : i64 to !llvm.ptr
    %intptr_44 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %66 = arith.index_cast %intptr_44 : index to i64
    %base_buffer_45, %offset_46, %sizes_47, %strides_48 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %67 = arith.index_cast %offset_46 : index to i64
    %c0_i64_49 = arith.constant 0 : i64
    %68 = arith.index_cast %strides_48 : index to i64
    %69 = arith.muli %c0_i64_49, %68 : i64
    %70 = arith.addi %67, %69 : i64
    %c8_i64_50 = arith.constant 8 : i64
    %71 = arith.muli %70, %c8_i64_50 : i64
    %72 = arith.addi %66, %71 : i64
    %73 = llvm.inttoptr %72 : i64 to !llvm.ptr
    %intptr_51 = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %74 = arith.index_cast %intptr_51 : index to i64
    %base_buffer_52, %offset_53, %sizes_54, %strides_55 = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %75 = arith.index_cast %offset_53 : index to i64
    %c0_i64_56 = arith.constant 0 : i64
    %76 = arith.index_cast %strides_55 : index to i64
    %77 = arith.muli %c0_i64_56, %76 : i64
    %78 = arith.addi %75, %77 : i64
    %c8_i64_57 = arith.constant 8 : i64
    %79 = arith.muli %78, %c8_i64_57 : i64
    %80 = arith.addi %74, %79 : i64
    %81 = llvm.inttoptr %80 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%53, %54, %cst_35, %65, %54, %73, %cst_35, %81) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_58 = memref.subview %arg4[0] [%13] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %82 = bufferization.to_tensor %subview_58 restrict writable : memref<?xf64, strided<[1]>>
    %83 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %84 = bufferization.to_memref %83 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %84, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

