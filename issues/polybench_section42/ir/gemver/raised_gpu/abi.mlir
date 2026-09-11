module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg6 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg7 restrict : memref<?xf64>
    %5 = bufferization.to_tensor %arg8 restrict : memref<?xf64>
    %6 = bufferization.to_tensor %arg9 restrict : memref<?xf64>
    %7 = bufferization.to_tensor %arg10 restrict : memref<?xf64>
    %8 = bufferization.to_tensor %arg11 restrict : memref<?xf64>
    %9 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %1[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %3[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_2 = tensor.extract_slice %4[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_3 = tensor.extract_slice %0[0, 0] [%9, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %subview = memref.subview %arg3[0, 0] [%9, %9] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_4 = memref.subview %arg4[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_5 = memref.subview %arg5[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_6 = memref.subview %arg6[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_7 = memref.subview %arg7[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %subview, %c0 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %10 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_8 = memref.dim %subview, %c1 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %11 = arith.index_cast %dim_8 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %12 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %13 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %14 = arith.muli %13, %c8_i64 : i64
    %15 = arith.addi %12, %14 : i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %subview_4 : memref<?xf64, strided<[1]>> -> index
    %17 = arith.index_cast %intptr_9 : index to i64
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %subview_4 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %18 = arith.index_cast %offset_11 : index to i64
    %c8_i64_14 = arith.constant 8 : i64
    %19 = arith.muli %18, %c8_i64_14 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %intptr_15 = memref.extract_aligned_pointer_as_index %subview_5 : memref<?xf64, strided<[1]>> -> index
    %22 = arith.index_cast %intptr_15 : index to i64
    %base_buffer_16, %offset_17, %sizes_18, %strides_19 = memref.extract_strided_metadata %subview_5 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %23 = arith.index_cast %offset_17 : index to i64
    %c8_i64_20 = arith.constant 8 : i64
    %24 = arith.muli %23, %c8_i64_20 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %intptr_21 = memref.extract_aligned_pointer_as_index %subview_6 : memref<?xf64, strided<[1]>> -> index
    %27 = arith.index_cast %intptr_21 : index to i64
    %base_buffer_22, %offset_23, %sizes_24, %strides_25 = memref.extract_strided_metadata %subview_6 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %28 = arith.index_cast %offset_23 : index to i64
    %c8_i64_26 = arith.constant 8 : i64
    %29 = arith.muli %28, %c8_i64_26 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %intptr_27 = memref.extract_aligned_pointer_as_index %subview_7 : memref<?xf64, strided<[1]>> -> index
    %32 = arith.index_cast %intptr_27 : index to i64
    %base_buffer_28, %offset_29, %sizes_30, %strides_31 = memref.extract_strided_metadata %subview_7 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %33 = arith.index_cast %offset_29 : index to i64
    %c8_i64_32 = arith.constant 8 : i64
    %34 = arith.muli %33, %c8_i64_32 : i64
    %35 = arith.addi %32, %34 : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dger_rank2(%10, %11, %21, %26, %31, %36, %16, %11) : (i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %37 = bufferization.to_tensor %subview restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice = tensor.insert_slice %37 into %0[0, 0] [%9, %9] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %38 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %38, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %7[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_34 = tensor.extract_slice %6[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %cst = arith.constant 1.000000e+00 : f64
    %subview_35 = memref.subview %arg10[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_36 = memref.subview %arg9[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c0_37 = arith.constant 0 : index
    %dim_38 = memref.dim %subview, %c0_37 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %39 = arith.index_cast %dim_38 : index to i32
    %c1_39 = arith.constant 1 : index
    %dim_40 = memref.dim %subview, %c1_39 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %40 = arith.index_cast %dim_40 : index to i32
    %intptr_41 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %41 = arith.index_cast %intptr_41 : index to i64
    %base_buffer_42, %offset_43, %sizes_44:2, %strides_45:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %42 = arith.index_cast %offset_43 : index to i64
    %c8_i64_46 = arith.constant 8 : i64
    %43 = arith.muli %42, %c8_i64_46 : i64
    %44 = arith.addi %41, %43 : i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_47 = memref.extract_aligned_pointer_as_index %subview_35 : memref<?xf64, strided<[1]>> -> index
    %46 = arith.index_cast %intptr_47 : index to i64
    %base_buffer_48, %offset_49, %sizes_50, %strides_51 = memref.extract_strided_metadata %subview_35 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %47 = arith.index_cast %offset_49 : index to i64
    %c8_i64_52 = arith.constant 8 : i64
    %48 = arith.muli %47, %c8_i64_52 : i64
    %49 = arith.addi %46, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %intptr_53 = memref.extract_aligned_pointer_as_index %subview_36 : memref<?xf64, strided<[1]>> -> index
    %51 = arith.index_cast %intptr_53 : index to i64
    %base_buffer_54, %offset_55, %sizes_56, %strides_57 = memref.extract_strided_metadata %subview_36 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %52 = arith.index_cast %offset_55 : index to i64
    %c8_i64_58 = arith.constant 8 : i64
    %53 = arith.muli %52, %c8_i64_58 : i64
    %54 = arith.addi %51, %53 : i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%39, %40, %arg2, %45, %40, %50, %cst, %55) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %56 = bufferization.to_tensor %subview_36 restrict writable : memref<?xf64, strided<[1]>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice_59 = tensor.insert_slice %56 into %6[0] [%9] [1] : tensor<?xf64> into tensor<?xf64>
    %cst_60 = arith.constant 1.000000e+00 : f64
    %cst_61 = arith.constant 1.000000e+00 : f64
    %57 = bufferization.to_memref %inserted_slice_59 : memref<?xf64>
    %c0_62 = arith.constant 0 : index
    %dim_63 = memref.dim %57, %c0_62 : memref<?xf64>
    %58 = arith.index_cast %dim_63 : index to i32
    %intptr_64 = memref.extract_aligned_pointer_as_index %arg11 : memref<?xf64> -> index
    %59 = arith.index_cast %intptr_64 : index to i64
    %base_buffer_65, %offset_66, %sizes_67, %strides_68 = memref.extract_strided_metadata %arg11 : memref<?xf64> -> memref<f64>, index, index, index
    %60 = arith.index_cast %offset_66 : index to i64
    %c8_i64_69 = arith.constant 8 : i64
    %61 = arith.muli %60, %c8_i64_69 : i64
    %62 = arith.addi %59, %61 : i64
    %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
    %intptr_70 = memref.extract_aligned_pointer_as_index %57 : memref<?xf64> -> index
    %64 = arith.index_cast %intptr_70 : index to i64
    %base_buffer_71, %offset_72, %sizes_73, %strides_74 = memref.extract_strided_metadata %57 : memref<?xf64> -> memref<f64>, index, index, index
    %65 = arith.index_cast %offset_72 : index to i64
    %c8_i64_75 = arith.constant 8 : i64
    %66 = arith.muli %65, %c8_i64_75 : i64
    %67 = arith.addi %64, %66 : i64
    %68 = llvm.inttoptr %67 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_daxpby(%58, %cst_60, %63, %cst_61, %68) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    %69 = bufferization.to_tensor %57 restrict writable : memref<?xf64>
    %70 = bufferization.to_memref %69 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %70, %arg9 : memref<?xf64> to memref<?xf64>
    %extracted_slice_76 = tensor.extract_slice %69[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_77 = tensor.extract_slice %5[0] [%9] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_78 = arith.constant 1.000000e+00 : f64
    %subview_79 = memref.subview %57[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_80 = memref.subview %arg8[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c0_81 = arith.constant 0 : index
    %dim_82 = memref.dim %subview, %c0_81 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %71 = arith.index_cast %dim_82 : index to i32
    %c1_83 = arith.constant 1 : index
    %dim_84 = memref.dim %subview, %c1_83 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %72 = arith.index_cast %dim_84 : index to i32
    %intptr_85 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %73 = arith.index_cast %intptr_85 : index to i64
    %base_buffer_86, %offset_87, %sizes_88:2, %strides_89:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %74 = arith.index_cast %offset_87 : index to i64
    %c8_i64_90 = arith.constant 8 : i64
    %75 = arith.muli %74, %c8_i64_90 : i64
    %76 = arith.addi %73, %75 : i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    %intptr_91 = memref.extract_aligned_pointer_as_index %subview_79 : memref<?xf64, strided<[1]>> -> index
    %78 = arith.index_cast %intptr_91 : index to i64
    %base_buffer_92, %offset_93, %sizes_94, %strides_95 = memref.extract_strided_metadata %subview_79 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %79 = arith.index_cast %offset_93 : index to i64
    %c8_i64_96 = arith.constant 8 : i64
    %80 = arith.muli %79, %c8_i64_96 : i64
    %81 = arith.addi %78, %80 : i64
    %82 = llvm.inttoptr %81 : i64 to !llvm.ptr
    %intptr_97 = memref.extract_aligned_pointer_as_index %subview_80 : memref<?xf64, strided<[1]>> -> index
    %83 = arith.index_cast %intptr_97 : index to i64
    %base_buffer_98, %offset_99, %sizes_100, %strides_101 = memref.extract_strided_metadata %subview_80 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %84 = arith.index_cast %offset_99 : index to i64
    %c8_i64_102 = arith.constant 8 : i64
    %85 = arith.muli %84, %c8_i64_102 : i64
    %86 = arith.addi %83, %85 : i64
    %87 = llvm.inttoptr %86 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%71, %72, %arg1, %77, %72, %82, %cst_78, %87) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %88 = bufferization.to_tensor %subview_80 restrict writable : memref<?xf64, strided<[1]>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %inserted_slice_103 = tensor.insert_slice %88 into %5[0] [%9] [1] : tensor<?xf64> into tensor<?xf64>
    %89 = bufferization.to_memref %inserted_slice_103 : memref<?xf64>
    memref.copy %89, %arg8 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_dger_rank2(i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

