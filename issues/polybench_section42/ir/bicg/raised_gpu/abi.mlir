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
    %c1 = arith.constant 1 : index
    %dim_11 = tensor.dim %0, %c1 : tensor<?x?xf64>
    %23 = arith.index_cast %dim_11 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_12 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %24 = arith.index_cast %intptr_12 : index to i64
    %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %25 = arith.index_cast %offset_14 : index to i64
    %c0_i64 = arith.constant 0 : i64
    %26 = arith.index_cast %strides_16#0 : index to i64
    %27 = arith.muli %c0_i64, %26 : i64
    %28 = arith.addi %25, %27 : i64
    %c0_i64_17 = arith.constant 0 : i64
    %29 = arith.index_cast %strides_16#1 : index to i64
    %30 = arith.muli %c0_i64_17, %29 : i64
    %31 = arith.addi %28, %30 : i64
    %c8_i64_18 = arith.constant 8 : i64
    %32 = arith.muli %31, %c8_i64_18 : i64
    %33 = arith.addi %24, %32 : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %intptr_19 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %35 = arith.index_cast %intptr_19 : index to i64
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %arg6 : memref<?xf64> -> memref<f64>, index, index, index
    %36 = arith.index_cast %offset_21 : index to i64
    %c0_i64_24 = arith.constant 0 : i64
    %37 = arith.index_cast %strides_23 : index to i64
    %38 = arith.muli %c0_i64_24, %37 : i64
    %39 = arith.addi %36, %38 : i64
    %c8_i64_25 = arith.constant 8 : i64
    %40 = arith.muli %39, %c8_i64_25 : i64
    %41 = arith.addi %35, %40 : i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    %subview = memref.subview %arg3[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_26 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %43 = arith.index_cast %intptr_26 : index to i64
    %base_buffer_27, %offset_28, %sizes_29, %strides_30 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %44 = arith.index_cast %offset_28 : index to i64
    %c8_i64_31 = arith.constant 8 : i64
    %45 = arith.muli %44, %c8_i64_31 : i64
    %46 = arith.addi %43, %45 : i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%21, %22, %cst_10, %34, %23, %42, %cst_10, %47) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %48 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %49 = bufferization.to_tensor %arg3 restrict writable : memref<?xf64>
    %50 = bufferization.to_memref %49 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %50, %arg3 : memref<?xf64> to memref<?xf64>
    %extracted_slice_32 = tensor.extract_slice %0[0, 0] [%13, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_34 = tensor.extract_slice %20[0] [%13] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_35 = arith.constant 1.000000e+00 : f64
    %51 = arith.index_cast %13 : index to i32
    %52 = arith.index_cast %5 : index to i32
    %c1_36 = arith.constant 1 : index
    %dim_37 = tensor.dim %0, %c1_36 : tensor<?x?xf64>
    %53 = arith.index_cast %dim_37 : index to i32
    %intptr_38 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %54 = arith.index_cast %intptr_38 : index to i64
    %base_buffer_39, %offset_40, %sizes_41:2, %strides_42:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %55 = arith.index_cast %offset_40 : index to i64
    %c0_i64_43 = arith.constant 0 : i64
    %56 = arith.index_cast %strides_42#0 : index to i64
    %57 = arith.muli %c0_i64_43, %56 : i64
    %58 = arith.addi %55, %57 : i64
    %c0_i64_44 = arith.constant 0 : i64
    %59 = arith.index_cast %strides_42#1 : index to i64
    %60 = arith.muli %c0_i64_44, %59 : i64
    %61 = arith.addi %58, %60 : i64
    %c8_i64_45 = arith.constant 8 : i64
    %62 = arith.muli %61, %c8_i64_45 : i64
    %63 = arith.addi %54, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %intptr_46 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %65 = arith.index_cast %intptr_46 : index to i64
    %base_buffer_47, %offset_48, %sizes_49, %strides_50 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %66 = arith.index_cast %offset_48 : index to i64
    %c0_i64_51 = arith.constant 0 : i64
    %67 = arith.index_cast %strides_50 : index to i64
    %68 = arith.muli %c0_i64_51, %67 : i64
    %69 = arith.addi %66, %68 : i64
    %c8_i64_52 = arith.constant 8 : i64
    %70 = arith.muli %69, %c8_i64_52 : i64
    %71 = arith.addi %65, %70 : i64
    %72 = llvm.inttoptr %71 : i64 to !llvm.ptr
    %subview_53 = memref.subview %arg4[0] [%13] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_54 = memref.extract_aligned_pointer_as_index %subview_53 : memref<?xf64, strided<[1]>> -> index
    %73 = arith.index_cast %intptr_54 : index to i64
    %base_buffer_55, %offset_56, %sizes_57, %strides_58 = memref.extract_strided_metadata %subview_53 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %74 = arith.index_cast %offset_56 : index to i64
    %c8_i64_59 = arith.constant 8 : i64
    %75 = arith.muli %74, %c8_i64_59 : i64
    %76 = arith.addi %73, %75 : i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%51, %52, %cst_35, %64, %53, %72, %cst_35, %77) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %78 = bufferization.to_tensor %subview_53 restrict writable : memref<?xf64, strided<[1]>>
    %79 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %80 = bufferization.to_memref %79 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %80, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

