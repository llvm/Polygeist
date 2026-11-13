module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg5, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg6, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg6, %c1_3 : memref<?x?xf64>
    %memref_5, %asyncToken_6 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    %2 = gpu.memcpy async [%asyncToken_6] %memref_5, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg7, %c0_7 : memref<?x?xf64>
    %c1_9 = arith.constant 1 : index
    %dim_10 = memref.dim %arg7, %c1_9 : memref<?x?xf64>
    %memref_11, %asyncToken_12 = gpu.alloc async [%2] (%dim_8, %dim_10) : memref<?x?xf64>
    %3 = gpu.memcpy async [%asyncToken_12] %memref_11, %arg7 : memref<?x?xf64>, memref<?x?xf64>
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %arg8, %c0_13 : memref<?x?xf64>
    %c1_15 = arith.constant 1 : index
    %dim_16 = memref.dim %arg8, %c1_15 : memref<?x?xf64>
    %memref_17, %asyncToken_18 = gpu.alloc async [%3] (%dim_14, %dim_16) : memref<?x?xf64>
    %4 = gpu.memcpy async [%asyncToken_18] %memref_17, %arg8 : memref<?x?xf64>, memref<?x?xf64>
    %c0_19 = arith.constant 0 : index
    %dim_20 = memref.dim %arg9, %c0_19 : memref<?x?xf64>
    %c1_21 = arith.constant 1 : index
    %dim_22 = memref.dim %arg9, %c1_21 : memref<?x?xf64>
    %memref_23, %asyncToken_24 = gpu.alloc async [%4] (%dim_20, %dim_22) : memref<?x?xf64>
    %5 = gpu.memcpy async [%asyncToken_24] %memref_23, %arg9 : memref<?x?xf64>, memref<?x?xf64>
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg10, %c0_25 : memref<?x?xf64>
    %c1_27 = arith.constant 1 : index
    %dim_28 = memref.dim %arg10, %c1_27 : memref<?x?xf64>
    %memref_29, %asyncToken_30 = gpu.alloc async [%5] (%dim_26, %dim_28) : memref<?x?xf64>
    %6 = gpu.memcpy async [%asyncToken_30] %memref_29, %arg10 : memref<?x?xf64>, memref<?x?xf64>
    %c0_31 = arith.constant 0 : index
    %dim_32 = memref.dim %arg11, %c0_31 : memref<?x?xf64>
    %c1_33 = arith.constant 1 : index
    %dim_34 = memref.dim %arg11, %c1_33 : memref<?x?xf64>
    %memref_35, %asyncToken_36 = gpu.alloc async [%6] (%dim_32, %dim_34) : memref<?x?xf64>
    %7 = gpu.memcpy async [%asyncToken_36] %memref_35, %arg11 : memref<?x?xf64>, memref<?x?xf64>
    gpu.wait [%7]
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %c1_37 = arith.constant 1 : index
    %c0_38 = arith.constant 0 : index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg2 : i32 to index
    %10 = arith.index_cast %arg4 : i32 to index
    %11 = arith.index_cast %arg3 : i32 to index
    %12 = arith.index_cast %arg0 : i32 to index
    %dim_39 = memref.dim %memref, %c0_38 : memref<?x?xf64>
    %13 = arith.index_cast %dim_39 : index to i32
    %dim_40 = memref.dim %memref, %c1_37 : memref<?x?xf64>
    %14 = arith.index_cast %dim_40 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%13, %14, %16, %14) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview = memref.subview %memref_5[0, 0] [%12, %9] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_41 = memref.subview %memref_11[0, 0] [%9, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_42 = memref.subview %memref[0, 0] [%12, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_43 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %17 = arith.index_cast %intptr_43 : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %18 = arith.index_cast %offset : index to i64
    %19 = arith.muli %18, %c8_i64 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %intptr_44 = memref.extract_aligned_pointer_as_index %subview_41 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %22 = arith.index_cast %intptr_44 : index to i64
    %base_buffer_45, %offset_46, %sizes_47:2, %strides_48:2 = memref.extract_strided_metadata %subview_41 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %23 = arith.index_cast %offset_46 : index to i64
    %24 = arith.muli %23, %c8_i64 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %intptr_49 = memref.extract_aligned_pointer_as_index %subview_42 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %27 = arith.index_cast %intptr_49 : index to i64
    %base_buffer_50, %offset_51, %sizes_52:2, %strides_53:2 = memref.extract_strided_metadata %subview_42 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %28 = arith.index_cast %offset_51 : index to i64
    %29 = arith.muli %28, %c8_i64 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%arg0, %arg1, %arg2, %cst, %21, %arg2, %26, %arg1, %cst, %31, %arg1) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %dim_54 = memref.dim %memref_17, %c0_38 : memref<?x?xf64>
    %32 = arith.index_cast %dim_54 : index to i32
    %dim_55 = memref.dim %memref_17, %c1_37 : memref<?x?xf64>
    %33 = arith.index_cast %dim_55 : index to i32
    %intptr_56 = memref.extract_aligned_pointer_as_index %memref_17 : memref<?x?xf64> -> index
    %34 = arith.index_cast %intptr_56 : index to i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%32, %33, %35, %33) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview_57 = memref.subview %memref_23[0, 0] [%8, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_58 = memref.subview %memref_29[0, 0] [%10, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_59 = memref.subview %memref_17[0, 0] [%8, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_60 = memref.extract_aligned_pointer_as_index %subview_57 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %36 = arith.index_cast %intptr_60 : index to i64
    %base_buffer_61, %offset_62, %sizes_63:2, %strides_64:2 = memref.extract_strided_metadata %subview_57 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %37 = arith.index_cast %offset_62 : index to i64
    %38 = arith.muli %37, %c8_i64 : i64
    %39 = arith.addi %36, %38 : i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %intptr_65 = memref.extract_aligned_pointer_as_index %subview_58 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %41 = arith.index_cast %intptr_65 : index to i64
    %base_buffer_66, %offset_67, %sizes_68:2, %strides_69:2 = memref.extract_strided_metadata %subview_58 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %42 = arith.index_cast %offset_67 : index to i64
    %43 = arith.muli %42, %c8_i64 : i64
    %44 = arith.addi %41, %43 : i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_70 = memref.extract_aligned_pointer_as_index %subview_59 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %46 = arith.index_cast %intptr_70 : index to i64
    %base_buffer_71, %offset_72, %sizes_73:2, %strides_74:2 = memref.extract_strided_metadata %subview_59 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %47 = arith.index_cast %offset_72 : index to i64
    %48 = arith.muli %47, %c8_i64 : i64
    %49 = arith.addi %46, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%arg1, %arg3, %arg4, %cst, %40, %arg4, %45, %arg3, %cst, %50, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %dim_75 = memref.dim %memref_35, %c0_38 : memref<?x?xf64>
    %51 = arith.index_cast %dim_75 : index to i32
    %dim_76 = memref.dim %memref_35, %c1_37 : memref<?x?xf64>
    %52 = arith.index_cast %dim_76 : index to i32
    %intptr_77 = memref.extract_aligned_pointer_as_index %memref_35 : memref<?x?xf64> -> index
    %53 = arith.index_cast %intptr_77 : index to i64
    %54 = llvm.inttoptr %53 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%51, %52, %54, %52) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview_78 = memref.subview %memref_35[0, 0] [%12, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_79 = memref.extract_aligned_pointer_as_index %subview_42 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %55 = arith.index_cast %intptr_79 : index to i64
    %base_buffer_80, %offset_81, %sizes_82:2, %strides_83:2 = memref.extract_strided_metadata %subview_42 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %56 = arith.index_cast %offset_81 : index to i64
    %57 = arith.muli %56, %c8_i64 : i64
    %58 = arith.addi %55, %57 : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr
    %intptr_84 = memref.extract_aligned_pointer_as_index %subview_59 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %60 = arith.index_cast %intptr_84 : index to i64
    %base_buffer_85, %offset_86, %sizes_87:2, %strides_88:2 = memref.extract_strided_metadata %subview_59 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %61 = arith.index_cast %offset_86 : index to i64
    %62 = arith.muli %61, %c8_i64 : i64
    %63 = arith.addi %60, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %intptr_89 = memref.extract_aligned_pointer_as_index %subview_78 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %65 = arith.index_cast %intptr_89 : index to i64
    %base_buffer_90, %offset_91, %sizes_92:2, %strides_93:2 = memref.extract_strided_metadata %subview_78 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %66 = arith.index_cast %offset_91 : index to i64
    %67 = arith.muli %66, %c8_i64 : i64
    %68 = arith.addi %65, %67 : i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%arg0, %arg3, %arg1, %cst, %59, %arg1, %64, %arg3, %cst, %69, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %70 = gpu.wait async
    %71 = gpu.memcpy async [%70] %arg5, %memref : memref<?x?xf64>, memref<?x?xf64>
    %72 = gpu.dealloc async [%71] %memref : memref<?x?xf64>
    %73 = gpu.memcpy async [%72] %arg6, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %74 = gpu.dealloc async [%73] %memref_5 : memref<?x?xf64>
    %75 = gpu.memcpy async [%74] %arg7, %memref_11 : memref<?x?xf64>, memref<?x?xf64>
    %76 = gpu.dealloc async [%75] %memref_11 : memref<?x?xf64>
    %77 = gpu.memcpy async [%76] %arg8, %memref_17 : memref<?x?xf64>, memref<?x?xf64>
    %78 = gpu.dealloc async [%77] %memref_17 : memref<?x?xf64>
    %79 = gpu.memcpy async [%78] %arg9, %memref_23 : memref<?x?xf64>, memref<?x?xf64>
    %80 = gpu.dealloc async [%79] %memref_23 : memref<?x?xf64>
    %81 = gpu.memcpy async [%80] %arg10, %memref_29 : memref<?x?xf64>, memref<?x?xf64>
    %82 = gpu.dealloc async [%81] %memref_29 : memref<?x?xf64>
    %83 = gpu.memcpy async [%82] %arg11, %memref_35 : memref<?x?xf64>, memref<?x?xf64>
    %84 = gpu.dealloc async [%83] %memref_35 : memref<?x?xf64>
    gpu.wait [%84]
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

