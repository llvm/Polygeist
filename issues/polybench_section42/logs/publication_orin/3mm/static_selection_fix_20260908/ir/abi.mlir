module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg7 : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg8 : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %5 = bufferization.to_tensor %arg10 : memref<?x?xf64>
    %6 = bufferization.to_tensor %arg11 : memref<?x?xf64>
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
    %cst_4 = arith.constant 0.000000e+00 : f64
    %subview = memref.subview %arg6[0, 0] [%11, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %arg7[0, 0] [%8, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_6 = memref.subview %arg5[0, 0] [%11, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %subview, %c0_7 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %20 = arith.index_cast %dim_8 : index to i32
    %c1_9 = arith.constant 1 : index
    %dim_10 = memref.dim %subview, %c1_9 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %21 = arith.index_cast %dim_10 : index to i32
    %c1_11 = arith.constant 1 : index
    %dim_12 = memref.dim %subview_5, %c1_11 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %22 = arith.index_cast %dim_12 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_13 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %23 = arith.index_cast %intptr_13 : index to i64
    %base_buffer_14, %offset_15, %sizes_16:2, %strides_17:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %24 = arith.index_cast %offset_15 : index to i64
    %c8_i64_18 = arith.constant 8 : i64
    %25 = arith.muli %24, %c8_i64_18 : i64
    %26 = arith.addi %23, %25 : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    %intptr_19 = memref.extract_aligned_pointer_as_index %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %28 = arith.index_cast %intptr_19 : index to i64
    %base_buffer_20, %offset_21, %sizes_22:2, %strides_23:2 = memref.extract_strided_metadata %subview_5 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %29 = arith.index_cast %offset_21 : index to i64
    %c8_i64_24 = arith.constant 8 : i64
    %30 = arith.muli %29, %c8_i64_24 : i64
    %31 = arith.addi %28, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %intptr_25 = memref.extract_aligned_pointer_as_index %subview_6 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %33 = arith.index_cast %intptr_25 : index to i64
    %base_buffer_26, %offset_27, %sizes_28:2, %strides_29:2 = memref.extract_strided_metadata %subview_6 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %34 = arith.index_cast %offset_27 : index to i64
    %c8_i64_30 = arith.constant 8 : i64
    %35 = arith.muli %34, %c8_i64_30 : i64
    %36 = arith.addi %33, %35 : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%20, %22, %21, %cst_3, %27, %21, %32, %22, %cst_4, %37, %22) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %38 = bufferization.to_tensor %subview_6 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice = tensor.insert_slice %38 into %19[0, 0] [%11, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %39 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    %c0_31 = arith.constant 0 : index
    %dim_32 = memref.dim %arg8, %c0_31 : memref<?x?xf64>
    %40 = arith.index_cast %dim_32 : index to i32
    %c1_33 = arith.constant 1 : index
    %dim_34 = memref.dim %arg8, %c1_33 : memref<?x?xf64>
    %41 = arith.index_cast %dim_34 : index to i32
    %intptr_35 = memref.extract_aligned_pointer_as_index %arg8 : memref<?x?xf64> -> index
    %42 = arith.index_cast %intptr_35 : index to i64
    %base_buffer_36, %offset_37, %sizes_38:2, %strides_39:2 = memref.extract_strided_metadata %arg8 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %43 = arith.index_cast %offset_37 : index to i64
    %c8_i64_40 = arith.constant 8 : i64
    %44 = arith.muli %43, %c8_i64_40 : i64
    %45 = arith.addi %42, %44 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%40, %41, %46, %41) : (i32, i32, !llvm.ptr, i32) -> ()
    %47 = bufferization.to_tensor %arg8 restrict writable : memref<?x?xf64>
    %extracted_slice_41 = tensor.extract_slice %4[0, 0] [%7, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_42 = tensor.extract_slice %5[0, 0] [%9, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_43 = tensor.extract_slice %47[0, 0] [%7, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_44 = arith.constant 1.000000e+00 : f64
    %cst_45 = arith.constant 0.000000e+00 : f64
    %subview_46 = memref.subview %arg9[0, 0] [%7, %9] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_47 = memref.subview %arg10[0, 0] [%9, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_48 = memref.subview %arg8[0, 0] [%7, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_49 = arith.constant 0 : index
    %dim_50 = memref.dim %subview_46, %c0_49 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %48 = arith.index_cast %dim_50 : index to i32
    %c1_51 = arith.constant 1 : index
    %dim_52 = memref.dim %subview_46, %c1_51 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %49 = arith.index_cast %dim_52 : index to i32
    %c1_53 = arith.constant 1 : index
    %dim_54 = memref.dim %subview_47, %c1_53 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %50 = arith.index_cast %dim_54 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_55 = memref.extract_aligned_pointer_as_index %subview_46 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %51 = arith.index_cast %intptr_55 : index to i64
    %base_buffer_56, %offset_57, %sizes_58:2, %strides_59:2 = memref.extract_strided_metadata %subview_46 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %52 = arith.index_cast %offset_57 : index to i64
    %c8_i64_60 = arith.constant 8 : i64
    %53 = arith.muli %52, %c8_i64_60 : i64
    %54 = arith.addi %51, %53 : i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    %intptr_61 = memref.extract_aligned_pointer_as_index %subview_47 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %56 = arith.index_cast %intptr_61 : index to i64
    %base_buffer_62, %offset_63, %sizes_64:2, %strides_65:2 = memref.extract_strided_metadata %subview_47 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %57 = arith.index_cast %offset_63 : index to i64
    %c8_i64_66 = arith.constant 8 : i64
    %58 = arith.muli %57, %c8_i64_66 : i64
    %59 = arith.addi %56, %58 : i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %intptr_67 = memref.extract_aligned_pointer_as_index %subview_48 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %61 = arith.index_cast %intptr_67 : index to i64
    %base_buffer_68, %offset_69, %sizes_70:2, %strides_71:2 = memref.extract_strided_metadata %subview_48 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %62 = arith.index_cast %offset_69 : index to i64
    %c8_i64_72 = arith.constant 8 : i64
    %63 = arith.muli %62, %c8_i64_72 : i64
    %64 = arith.addi %61, %63 : i64
    %65 = llvm.inttoptr %64 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%48, %50, %49, %cst_44, %55, %49, %60, %50, %cst_45, %65, %50) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %66 = bufferization.to_tensor %subview_48 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice_73 = tensor.insert_slice %66 into %47[0, 0] [%7, %10] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %67 = bufferization.to_memref %inserted_slice_73 : memref<?x?xf64>
    %c0_74 = arith.constant 0 : index
    %dim_75 = memref.dim %arg11, %c0_74 : memref<?x?xf64>
    %68 = arith.index_cast %dim_75 : index to i32
    %c1_76 = arith.constant 1 : index
    %dim_77 = memref.dim %arg11, %c1_76 : memref<?x?xf64>
    %69 = arith.index_cast %dim_77 : index to i32
    %intptr_78 = memref.extract_aligned_pointer_as_index %arg11 : memref<?x?xf64> -> index
    %70 = arith.index_cast %intptr_78 : index to i64
    %base_buffer_79, %offset_80, %sizes_81:2, %strides_82:2 = memref.extract_strided_metadata %arg11 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %71 = arith.index_cast %offset_80 : index to i64
    %c8_i64_83 = arith.constant 8 : i64
    %72 = arith.muli %71, %c8_i64_83 : i64
    %73 = arith.addi %70, %72 : i64
    %74 = llvm.inttoptr %73 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%68, %69, %74, %69) : (i32, i32, !llvm.ptr, i32) -> ()
    %75 = bufferization.to_tensor %arg11 restrict writable : memref<?x?xf64>
    %extracted_slice_84 = tensor.extract_slice %75[0, 0] [%11, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %cst_85 = arith.constant 1.000000e+00 : f64
    %cst_86 = arith.constant 0.000000e+00 : f64
    %subview_87 = memref.subview %arg11[0, 0] [%11, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0_88 = arith.constant 0 : index
    %dim_89 = memref.dim %subview_6, %c0_88 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %76 = arith.index_cast %dim_89 : index to i32
    %c1_90 = arith.constant 1 : index
    %dim_91 = memref.dim %subview_6, %c1_90 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %77 = arith.index_cast %dim_91 : index to i32
    %c1_92 = arith.constant 1 : index
    %dim_93 = memref.dim %subview_48, %c1_92 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %78 = arith.index_cast %dim_93 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_94 = memref.extract_aligned_pointer_as_index %subview_6 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %79 = arith.index_cast %intptr_94 : index to i64
    %base_buffer_95, %offset_96, %sizes_97:2, %strides_98:2 = memref.extract_strided_metadata %subview_6 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %80 = arith.index_cast %offset_96 : index to i64
    %c8_i64_99 = arith.constant 8 : i64
    %81 = arith.muli %80, %c8_i64_99 : i64
    %82 = arith.addi %79, %81 : i64
    %83 = llvm.inttoptr %82 : i64 to !llvm.ptr
    %intptr_100 = memref.extract_aligned_pointer_as_index %subview_48 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %84 = arith.index_cast %intptr_100 : index to i64
    %base_buffer_101, %offset_102, %sizes_103:2, %strides_104:2 = memref.extract_strided_metadata %subview_48 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %85 = arith.index_cast %offset_102 : index to i64
    %c8_i64_105 = arith.constant 8 : i64
    %86 = arith.muli %85, %c8_i64_105 : i64
    %87 = arith.addi %84, %86 : i64
    %88 = llvm.inttoptr %87 : i64 to !llvm.ptr
    %intptr_106 = memref.extract_aligned_pointer_as_index %subview_87 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %89 = arith.index_cast %intptr_106 : index to i64
    %base_buffer_107, %offset_108, %sizes_109:2, %strides_110:2 = memref.extract_strided_metadata %subview_87 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %90 = arith.index_cast %offset_108 : index to i64
    %c8_i64_111 = arith.constant 8 : i64
    %91 = arith.muli %90, %c8_i64_111 : i64
    %92 = arith.addi %89, %91 : i64
    %93 = llvm.inttoptr %92 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%76, %78, %77, %cst_85, %83, %77, %88, %78, %cst_86, %93, %78) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %94 = bufferization.to_tensor %subview_87 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice_112 = tensor.insert_slice %94 into %75[0, 0] [%11, %10] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %95 = bufferization.to_memref %inserted_slice_112 : memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

