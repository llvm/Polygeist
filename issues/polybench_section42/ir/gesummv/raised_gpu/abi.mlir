module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg6 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg7 restrict : memref<?xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg5, %c0 : memref<?xf64>
    %6 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %7 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %8 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = arith.addi %7, %9 : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%6, %11) : (i32, !llvm.ptr) -> ()
    %12 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg7, %c0_0 : memref<?xf64>
    %13 = arith.index_cast %dim_1 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg7 : memref<?xf64> -> index
    %14 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg7 : memref<?xf64> -> memref<f64>, index, index, index
    %15 = arith.index_cast %offset_4 : index to i64
    %c8_i64_7 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64_7 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%13, %18) : (i32, !llvm.ptr) -> ()
    %19 = bufferization.to_tensor %arg7 restrict writable : memref<?xf64>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_8 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_9 = tensor.extract_slice %12[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_10 = arith.constant 1.000000e+00 : f64
    %20 = arith.index_cast %5 : index to i32
    %21 = arith.index_cast %5 : index to i32
    %c1 = arith.constant 1 : index
    %dim_11 = tensor.dim %0, %c1 : tensor<?x?xf64>
    %22 = arith.index_cast %dim_11 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_12 = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?xf64> -> index
    %23 = arith.index_cast %intptr_12 : index to i64
    %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
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
    %intptr_19 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %34 = arith.index_cast %intptr_19 : index to i64
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %arg6 : memref<?xf64> -> memref<f64>, index, index, index
    %35 = arith.index_cast %offset_21 : index to i64
    %c0_i64_24 = arith.constant 0 : i64
    %36 = arith.index_cast %strides_23 : index to i64
    %37 = arith.muli %c0_i64_24, %36 : i64
    %38 = arith.addi %35, %37 : i64
    %c8_i64_25 = arith.constant 8 : i64
    %39 = arith.muli %38, %c8_i64_25 : i64
    %40 = arith.addi %34, %39 : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %subview = memref.subview %arg5[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
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
    %extracted_slice_32 = tensor.extract_slice %1[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_34 = tensor.extract_slice %19[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_35 = arith.constant 1.000000e+00 : f64
    %50 = arith.index_cast %5 : index to i32
    %51 = arith.index_cast %5 : index to i32
    %c1_36 = arith.constant 1 : index
    %dim_37 = tensor.dim %1, %c1_36 : tensor<?x?xf64>
    %52 = arith.index_cast %dim_37 : index to i32
    %intptr_38 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %53 = arith.index_cast %intptr_38 : index to i64
    %base_buffer_39, %offset_40, %sizes_41:2, %strides_42:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %54 = arith.index_cast %offset_40 : index to i64
    %c0_i64_43 = arith.constant 0 : i64
    %55 = arith.index_cast %strides_42#0 : index to i64
    %56 = arith.muli %c0_i64_43, %55 : i64
    %57 = arith.addi %54, %56 : i64
    %c0_i64_44 = arith.constant 0 : i64
    %58 = arith.index_cast %strides_42#1 : index to i64
    %59 = arith.muli %c0_i64_44, %58 : i64
    %60 = arith.addi %57, %59 : i64
    %c8_i64_45 = arith.constant 8 : i64
    %61 = arith.muli %60, %c8_i64_45 : i64
    %62 = arith.addi %53, %61 : i64
    %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
    %intptr_46 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %64 = arith.index_cast %intptr_46 : index to i64
    %base_buffer_47, %offset_48, %sizes_49, %strides_50 = memref.extract_strided_metadata %arg6 : memref<?xf64> -> memref<f64>, index, index, index
    %65 = arith.index_cast %offset_48 : index to i64
    %c0_i64_51 = arith.constant 0 : i64
    %66 = arith.index_cast %strides_50 : index to i64
    %67 = arith.muli %c0_i64_51, %66 : i64
    %68 = arith.addi %65, %67 : i64
    %c8_i64_52 = arith.constant 8 : i64
    %69 = arith.muli %68, %c8_i64_52 : i64
    %70 = arith.addi %64, %69 : i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    %subview_53 = memref.subview %arg7[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_54 = memref.extract_aligned_pointer_as_index %subview_53 : memref<?xf64, strided<[1]>> -> index
    %72 = arith.index_cast %intptr_54 : index to i64
    %base_buffer_55, %offset_56, %sizes_57, %strides_58 = memref.extract_strided_metadata %subview_53 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %73 = arith.index_cast %offset_56 : index to i64
    %c8_i64_59 = arith.constant 8 : i64
    %74 = arith.muli %73, %c8_i64_59 : i64
    %75 = arith.addi %72, %74 : i64
    %76 = llvm.inttoptr %75 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%50, %51, %cst_35, %63, %52, %71, %cst_35, %76) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %77 = bufferization.to_tensor %subview_53 restrict writable : memref<?xf64, strided<[1]>>
    %78 = bufferization.to_tensor %arg7 restrict writable : memref<?xf64>
    %c0_60 = arith.constant 0 : index
    %dim_61 = memref.dim %arg7, %c0_60 : memref<?xf64>
    %79 = arith.index_cast %dim_61 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_62 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %80 = arith.index_cast %intptr_62 : index to i64
    %base_buffer_63, %offset_64, %sizes_65, %strides_66 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %81 = arith.index_cast %offset_64 : index to i64
    %c8_i64_67 = arith.constant 8 : i64
    %82 = arith.muli %81, %c8_i64_67 : i64
    %83 = arith.addi %80, %82 : i64
    %84 = llvm.inttoptr %83 : i64 to !llvm.ptr
    %intptr_68 = memref.extract_aligned_pointer_as_index %arg7 : memref<?xf64> -> index
    %85 = arith.index_cast %intptr_68 : index to i64
    %base_buffer_69, %offset_70, %sizes_71, %strides_72 = memref.extract_strided_metadata %arg7 : memref<?xf64> -> memref<f64>, index, index, index
    %86 = arith.index_cast %offset_70 : index to i64
    %c8_i64_73 = arith.constant 8 : i64
    %87 = arith.muli %86, %c8_i64_73 : i64
    %88 = arith.addi %85, %87 : i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_daxpby(%79, %arg1, %84, %arg2, %89) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    %90 = bufferization.to_tensor %arg7 restrict writable : memref<?xf64>
    %91 = bufferization.to_memref %90 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %91, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

