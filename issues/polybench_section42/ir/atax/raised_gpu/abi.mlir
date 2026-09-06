module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg4, %c0 : memref<?xf64>
    %5 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %7 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %8 = arith.muli %7, %c8_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%5, %10) : (i32, !llvm.ptr) -> ()
    %11 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %12 = arith.index_cast %arg0 : i32 to index
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg5, %c0_0 : memref<?xf64>
    %13 = arith.index_cast %dim_1 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %14 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %15 = arith.index_cast %offset_4 : index to i64
    %c8_i64_7 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64_7 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%13, %18) : (i32, !llvm.ptr) -> ()
    %19 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%12, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_8 = tensor.extract_slice %1[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_9 = tensor.extract_slice %19[0] [%12] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_10 = arith.constant 1.000000e+00 : f64
    %20 = arith.index_cast %12 : index to i32
    %21 = arith.index_cast %4 : index to i32
    %c1 = arith.constant 1 : index
    %dim_11 = tensor.dim %0, %c1 : tensor<?x?xf64>
    %22 = arith.index_cast %dim_11 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_12 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %23 = arith.index_cast %intptr_12 : index to i64
    %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %24 = arith.index_cast %offset_14 : index to i64
    %c0_i64 = arith.constant 0 : i64
    %25 = arith.index_cast %strides_16#0 : index to i64
    %26 = arith.muli %c0_i64, %25 : i64
    %27 = arith.addi %24, %26 : i64
    %c0_i64_17 = arith.constant 0 : i64
    %28 = arith.index_cast %strides_16#1 : index to i64
    %29 = arith.muli %c0_i64_17, %28 : i64
    %30 = arith.addi %27, %29 : i64
    %c8_i64_18 = arith.constant 8 : i64
    %31 = arith.muli %30, %c8_i64_18 : i64
    %32 = arith.addi %23, %31 : i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %intptr_19 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %34 = arith.index_cast %intptr_19 : index to i64
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
    %35 = arith.index_cast %offset_21 : index to i64
    %c0_i64_24 = arith.constant 0 : i64
    %36 = arith.index_cast %strides_23 : index to i64
    %37 = arith.muli %c0_i64_24, %36 : i64
    %38 = arith.addi %35, %37 : i64
    %c8_i64_25 = arith.constant 8 : i64
    %39 = arith.muli %38, %c8_i64_25 : i64
    %40 = arith.addi %34, %39 : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %subview = memref.subview %arg5[0] [%12] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_26 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %42 = arith.index_cast %intptr_26 : index to i64
    %base_buffer_27, %offset_28, %sizes_29, %strides_30 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %43 = arith.index_cast %offset_28 : index to i64
    %c8_i64_31 = arith.constant 8 : i64
    %44 = arith.muli %43, %c8_i64_31 : i64
    %45 = arith.addi %42, %44 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%20, %21, %cst_10, %33, %22, %41, %cst_10, %46) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %47 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %48 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    %49 = bufferization.to_memref %48 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %49, %arg5 : memref<?xf64> to memref<?xf64>
    %extracted_slice_32 = tensor.extract_slice %0[0, 0] [%12, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %11[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_34 = arith.constant 1.000000e+00 : f64
    %50 = arith.index_cast %12 : index to i32
    %51 = arith.index_cast %4 : index to i32
    %c1_35 = arith.constant 1 : index
    %dim_36 = tensor.dim %0, %c1_35 : tensor<?x?xf64>
    %52 = arith.index_cast %dim_36 : index to i32
    %intptr_37 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %53 = arith.index_cast %intptr_37 : index to i64
    %base_buffer_38, %offset_39, %sizes_40:2, %strides_41:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %54 = arith.index_cast %offset_39 : index to i64
    %c0_i64_42 = arith.constant 0 : i64
    %55 = arith.index_cast %strides_41#0 : index to i64
    %56 = arith.muli %c0_i64_42, %55 : i64
    %57 = arith.addi %54, %56 : i64
    %c0_i64_43 = arith.constant 0 : i64
    %58 = arith.index_cast %strides_41#1 : index to i64
    %59 = arith.muli %c0_i64_43, %58 : i64
    %60 = arith.addi %57, %59 : i64
    %c8_i64_44 = arith.constant 8 : i64
    %61 = arith.muli %60, %c8_i64_44 : i64
    %62 = arith.addi %53, %61 : i64
    %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
    %intptr_45 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %64 = arith.index_cast %intptr_45 : index to i64
    %base_buffer_46, %offset_47, %sizes_48, %strides_49 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %65 = arith.index_cast %offset_47 : index to i64
    %c8_i64_50 = arith.constant 8 : i64
    %66 = arith.muli %65, %c8_i64_50 : i64
    %67 = arith.addi %64, %66 : i64
    %68 = llvm.inttoptr %67 : i64 to !llvm.ptr
    %subview_51 = memref.subview %arg4[0] [%4] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_52 = memref.extract_aligned_pointer_as_index %subview_51 : memref<?xf64, strided<[1]>> -> index
    %69 = arith.index_cast %intptr_52 : index to i64
    %base_buffer_53, %offset_54, %sizes_55, %strides_56 = memref.extract_strided_metadata %subview_51 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %70 = arith.index_cast %offset_54 : index to i64
    %c8_i64_57 = arith.constant 8 : i64
    %71 = arith.muli %70, %c8_i64_57 : i64
    %72 = arith.addi %69, %71 : i64
    %73 = llvm.inttoptr %72 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%50, %51, %cst_34, %63, %52, %68, %cst_34, %73) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %74 = bufferization.to_tensor %subview_51 restrict writable : memref<?xf64, strided<[1]>>
    %75 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %76 = bufferization.to_memref %75 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %76, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

