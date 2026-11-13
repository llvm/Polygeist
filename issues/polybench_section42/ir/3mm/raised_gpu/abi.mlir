module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg7 restrict : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg8 restrict : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg9 restrict : memref<?x?xf64>
    %5 = bufferization.to_tensor %arg10 restrict : memref<?x?xf64>
    %6 = bufferization.to_tensor %arg11 restrict : memref<?x?xf64>
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg4 : i32 to index
    %10 = arith.index_cast %arg3 : i32 to index
    %11 = arith.index_cast %arg0 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg5, %c0 : memref<?x?xf64>
    %12 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %13 = arith.index_cast %dim_0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %14 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %15 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%12, %13, %18, %13) : (i32, i32, !llvm.ptr, i32) -> ()
    %19 = bufferization.to_tensor %arg5 restrict writable : memref<?x?xf64>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%11, %8] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %2[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %19[0, 0] [%11, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_3 = arith.constant 1.000000e+00 : f64
    %subview = memref.subview %arg6[0, 0] [%11, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_4 = memref.subview %arg7[0, 0] [%8, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %arg5[0, 0] [%11, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_6 = arith.constant 0 : index
    %dim_7 = memref.dim %subview, %c0_6 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %20 = arith.index_cast %dim_7 : index to i32
    %c1_8 = arith.constant 1 : index
    %dim_9 = memref.dim %subview, %c1_8 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %21 = arith.index_cast %dim_9 : index to i32
    %c1_10 = arith.constant 1 : index
    %dim_11 = memref.dim %subview_4, %c1_10 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %22 = arith.index_cast %dim_11 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_12 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %23 = arith.index_cast %intptr_12 : index to i64
    %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %24 = arith.index_cast %offset_14 : index to i64
    %c8_i64_17 = arith.constant 8 : i64
    %25 = arith.muli %24, %c8_i64_17 : i64
    %26 = arith.addi %23, %25 : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %28 = arith.index_cast %intptr_18 : index to i64
    %base_buffer_19, %offset_20, %sizes_21:2, %strides_22:2 = memref.extract_strided_metadata %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %29 = arith.index_cast %offset_20 : index to i64
    %c8_i64_23 = arith.constant 8 : i64
    %30 = arith.muli %29, %c8_i64_23 : i64
    %31 = arith.addi %28, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %intptr_24 = memref.extract_aligned_pointer_as_index %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %33 = arith.index_cast %intptr_24 : index to i64
    %base_buffer_25, %offset_26, %sizes_27:2, %strides_28:2 = memref.extract_strided_metadata %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %34 = arith.index_cast %offset_26 : index to i64
    %c8_i64_29 = arith.constant 8 : i64
    %35 = arith.muli %34, %c8_i64_29 : i64
    %36 = arith.addi %33, %35 : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%20, %22, %21, %cst_3, %27, %21, %32, %22, %cst_3, %37, %22) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %38 = bufferization.to_tensor %subview_5 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice = tensor.insert_slice %38 into %19[0, 0] [%11, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %39 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %39, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %c0_30 = arith.constant 0 : index
    %dim_31 = memref.dim %arg8, %c0_30 : memref<?x?xf64>
    %40 = arith.index_cast %dim_31 : index to i32
    %c1_32 = arith.constant 1 : index
    %dim_33 = memref.dim %arg8, %c1_32 : memref<?x?xf64>
    %41 = arith.index_cast %dim_33 : index to i32
    %intptr_34 = memref.extract_aligned_pointer_as_index %arg8 : memref<?x?xf64> -> index
    %42 = arith.index_cast %intptr_34 : index to i64
    %base_buffer_35, %offset_36, %sizes_37:2, %strides_38:2 = memref.extract_strided_metadata %arg8 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %43 = arith.index_cast %offset_36 : index to i64
    %c8_i64_39 = arith.constant 8 : i64
    %44 = arith.muli %43, %c8_i64_39 : i64
    %45 = arith.addi %42, %44 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%40, %41, %46, %41) : (i32, i32, !llvm.ptr, i32) -> ()
    %47 = bufferization.to_tensor %arg8 restrict writable : memref<?x?xf64>
    %extracted_slice_40 = tensor.extract_slice %4[0, 0] [%7, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_41 = tensor.extract_slice %5[0, 0] [%9, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_42 = tensor.extract_slice %47[0, 0] [%7, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_43 = arith.constant 1.000000e+00 : f64
    %subview_44 = memref.subview %arg9[0, 0] [%7, %9] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_45 = memref.subview %arg10[0, 0] [%9, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_46 = memref.subview %arg8[0, 0] [%7, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_47 = arith.constant 0 : index
    %dim_48 = memref.dim %subview_44, %c0_47 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %48 = arith.index_cast %dim_48 : index to i32
    %c1_49 = arith.constant 1 : index
    %dim_50 = memref.dim %subview_44, %c1_49 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %49 = arith.index_cast %dim_50 : index to i32
    %c1_51 = arith.constant 1 : index
    %dim_52 = memref.dim %subview_45, %c1_51 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %50 = arith.index_cast %dim_52 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_53 = memref.extract_aligned_pointer_as_index %subview_44 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %51 = arith.index_cast %intptr_53 : index to i64
    %base_buffer_54, %offset_55, %sizes_56:2, %strides_57:2 = memref.extract_strided_metadata %subview_44 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %52 = arith.index_cast %offset_55 : index to i64
    %c8_i64_58 = arith.constant 8 : i64
    %53 = arith.muli %52, %c8_i64_58 : i64
    %54 = arith.addi %51, %53 : i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    %intptr_59 = memref.extract_aligned_pointer_as_index %subview_45 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %56 = arith.index_cast %intptr_59 : index to i64
    %base_buffer_60, %offset_61, %sizes_62:2, %strides_63:2 = memref.extract_strided_metadata %subview_45 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %57 = arith.index_cast %offset_61 : index to i64
    %c8_i64_64 = arith.constant 8 : i64
    %58 = arith.muli %57, %c8_i64_64 : i64
    %59 = arith.addi %56, %58 : i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %intptr_65 = memref.extract_aligned_pointer_as_index %subview_46 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %61 = arith.index_cast %intptr_65 : index to i64
    %base_buffer_66, %offset_67, %sizes_68:2, %strides_69:2 = memref.extract_strided_metadata %subview_46 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %62 = arith.index_cast %offset_67 : index to i64
    %c8_i64_70 = arith.constant 8 : i64
    %63 = arith.muli %62, %c8_i64_70 : i64
    %64 = arith.addi %61, %63 : i64
    %65 = llvm.inttoptr %64 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%48, %50, %49, %cst_43, %55, %49, %60, %50, %cst_43, %65, %50) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %66 = bufferization.to_tensor %subview_46 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice_71 = tensor.insert_slice %66 into %47[0, 0] [%7, %10] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %67 = bufferization.to_memref %inserted_slice_71 : memref<?x?xf64>
    memref.copy %67, %arg8 : memref<?x?xf64> to memref<?x?xf64>
    %c0_72 = arith.constant 0 : index
    %dim_73 = memref.dim %arg11, %c0_72 : memref<?x?xf64>
    %68 = arith.index_cast %dim_73 : index to i32
    %c1_74 = arith.constant 1 : index
    %dim_75 = memref.dim %arg11, %c1_74 : memref<?x?xf64>
    %69 = arith.index_cast %dim_75 : index to i32
    %intptr_76 = memref.extract_aligned_pointer_as_index %arg11 : memref<?x?xf64> -> index
    %70 = arith.index_cast %intptr_76 : index to i64
    %base_buffer_77, %offset_78, %sizes_79:2, %strides_80:2 = memref.extract_strided_metadata %arg11 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %71 = arith.index_cast %offset_78 : index to i64
    %c8_i64_81 = arith.constant 8 : i64
    %72 = arith.muli %71, %c8_i64_81 : i64
    %73 = arith.addi %70, %72 : i64
    %74 = llvm.inttoptr %73 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%68, %69, %74, %69) : (i32, i32, !llvm.ptr, i32) -> ()
    %75 = bufferization.to_tensor %arg11 restrict writable : memref<?x?xf64>
    %extracted_slice_82 = tensor.extract_slice %75[0, 0] [%11, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_83 = arith.constant 1.000000e+00 : f64
    %subview_84 = memref.subview %arg11[0, 0] [%11, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_85 = arith.constant 0 : index
    %dim_86 = memref.dim %subview_5, %c0_85 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %76 = arith.index_cast %dim_86 : index to i32
    %c1_87 = arith.constant 1 : index
    %dim_88 = memref.dim %subview_5, %c1_87 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %77 = arith.index_cast %dim_88 : index to i32
    %c1_89 = arith.constant 1 : index
    %dim_90 = memref.dim %subview_46, %c1_89 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %78 = arith.index_cast %dim_90 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_91 = memref.extract_aligned_pointer_as_index %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %79 = arith.index_cast %intptr_91 : index to i64
    %base_buffer_92, %offset_93, %sizes_94:2, %strides_95:2 = memref.extract_strided_metadata %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %80 = arith.index_cast %offset_93 : index to i64
    %c8_i64_96 = arith.constant 8 : i64
    %81 = arith.muli %80, %c8_i64_96 : i64
    %82 = arith.addi %79, %81 : i64
    %83 = llvm.inttoptr %82 : i64 to !llvm.ptr
    %intptr_97 = memref.extract_aligned_pointer_as_index %subview_46 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %84 = arith.index_cast %intptr_97 : index to i64
    %base_buffer_98, %offset_99, %sizes_100:2, %strides_101:2 = memref.extract_strided_metadata %subview_46 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %85 = arith.index_cast %offset_99 : index to i64
    %c8_i64_102 = arith.constant 8 : i64
    %86 = arith.muli %85, %c8_i64_102 : i64
    %87 = arith.addi %84, %86 : i64
    %88 = llvm.inttoptr %87 : i64 to !llvm.ptr
    %intptr_103 = memref.extract_aligned_pointer_as_index %subview_84 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %89 = arith.index_cast %intptr_103 : index to i64
    %base_buffer_104, %offset_105, %sizes_106:2, %strides_107:2 = memref.extract_strided_metadata %subview_84 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %90 = arith.index_cast %offset_105 : index to i64
    %c8_i64_108 = arith.constant 8 : i64
    %91 = arith.muli %90, %c8_i64_108 : i64
    %92 = arith.addi %89, %91 : i64
    %93 = llvm.inttoptr %92 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%76, %78, %77, %cst_83, %83, %77, %88, %78, %cst_83, %93, %78) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %94 = bufferization.to_tensor %subview_84 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice_109 = tensor.insert_slice %94 into %75[0, 0] [%11, %10] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %95 = bufferization.to_memref %inserted_slice_109 : memref<?x?xf64>
    memref.copy %95, %arg11 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

