module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -1890640588580836432 : i64, polygeist.pipeline_scope_coalesced} {
    %c-1890640588580836432_i64 = arith.constant -1890640588580836432 : i64
    call @polygeist_gpu_region_timing_begin(%c-1890640588580836432_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg5, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg6, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg6, %c1_3 : memref<?x?xf64>
    %c2_i32_5 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_5) : (i32) -> ()
    %memref_6, %asyncToken_7 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_8 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_8) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_7] %memref_6, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg7, %c0_9 : memref<?x?xf64>
    %c1_11 = arith.constant 1 : index
    %dim_12 = memref.dim %arg7, %c1_11 : memref<?x?xf64>
    %c2_i32_13 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_13) : (i32) -> ()
    %memref_14, %asyncToken_15 = gpu.alloc async [%2] (%dim_10, %dim_12) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_16 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_16) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_15] %memref_14, %arg7 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_17 = arith.constant 0 : index
    %dim_18 = memref.dim %arg8, %c0_17 : memref<?x?xf64>
    %c1_19 = arith.constant 1 : index
    %dim_20 = memref.dim %arg8, %c1_19 : memref<?x?xf64>
    %c2_i32_21 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_21) : (i32) -> ()
    %memref_22, %asyncToken_23 = gpu.alloc async [%3] (%dim_18, %dim_20) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_24 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_24) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_23] %memref_22, %arg8 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg9, %c0_25 : memref<?x?xf64>
    %c1_27 = arith.constant 1 : index
    %dim_28 = memref.dim %arg9, %c1_27 : memref<?x?xf64>
    %c2_i32_29 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_29) : (i32) -> ()
    %memref_30, %asyncToken_31 = gpu.alloc async [%4] (%dim_26, %dim_28) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_32) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_31] %memref_30, %arg9 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_33 = arith.constant 0 : index
    %dim_34 = memref.dim %arg10, %c0_33 : memref<?x?xf64>
    %c1_35 = arith.constant 1 : index
    %dim_36 = memref.dim %arg10, %c1_35 : memref<?x?xf64>
    %c2_i32_37 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_37) : (i32) -> ()
    %memref_38, %asyncToken_39 = gpu.alloc async [%5] (%dim_34, %dim_36) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_40 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_40) : (i32) -> ()
    %6 = gpu.memcpy async [%asyncToken_39] %memref_38, %arg10 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_41 = arith.constant 0 : index
    %dim_42 = memref.dim %arg11, %c0_41 : memref<?x?xf64>
    %c1_43 = arith.constant 1 : index
    %dim_44 = memref.dim %arg11, %c1_43 : memref<?x?xf64>
    %c2_i32_45 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_45) : (i32) -> ()
    %memref_46, %asyncToken_47 = gpu.alloc async [%6] (%dim_42, %dim_44) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_48 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_48) : (i32) -> ()
    %7 = gpu.memcpy async [%asyncToken_47] %memref_46, %arg11 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%7]
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %c1_49 = arith.constant 1 : index
    %c0_50 = arith.constant 0 : index
    %cst_51 = arith.constant 0.000000e+00 : f64
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg2 : i32 to index
    %10 = arith.index_cast %arg4 : i32 to index
    %11 = arith.index_cast %arg3 : i32 to index
    %12 = arith.index_cast %arg0 : i32 to index
    %dim_52 = memref.dim %memref, %c0_50 : memref<?x?xf64>
    %13 = arith.index_cast %dim_52 : index to i32
    %dim_53 = memref.dim %memref, %c1_49 : memref<?x?xf64>
    %14 = arith.index_cast %dim_53 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%13, %14, %16, %14) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview = memref.subview %memref_6[0, 0] [%12, %9] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_54 = memref.subview %memref_14[0, 0] [%9, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_55 = memref.subview %memref[0, 0] [%12, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_56 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %17 = arith.index_cast %intptr_56 : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %18 = arith.index_cast %offset : index to i64
    %19 = arith.muli %18, %c8_i64 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %intptr_57 = memref.extract_aligned_pointer_as_index %subview_54 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %22 = arith.index_cast %intptr_57 : index to i64
    %base_buffer_58, %offset_59, %sizes_60:2, %strides_61:2 = memref.extract_strided_metadata %subview_54 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %23 = arith.index_cast %offset_59 : index to i64
    %24 = arith.muli %23, %c8_i64 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %intptr_62 = memref.extract_aligned_pointer_as_index %subview_55 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %27 = arith.index_cast %intptr_62 : index to i64
    %base_buffer_63, %offset_64, %sizes_65:2, %strides_66:2 = memref.extract_strided_metadata %subview_55 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %28 = arith.index_cast %offset_64 : index to i64
    %29 = arith.muli %28, %c8_i64 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg1, %arg2, %cst, %21, %arg2, %26, %arg1, %cst_51, %31, %arg1) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %dim_67 = memref.dim %memref_22, %c0_50 : memref<?x?xf64>
    %32 = arith.index_cast %dim_67 : index to i32
    %dim_68 = memref.dim %memref_22, %c1_49 : memref<?x?xf64>
    %33 = arith.index_cast %dim_68 : index to i32
    %intptr_69 = memref.extract_aligned_pointer_as_index %memref_22 : memref<?x?xf64> -> index
    %34 = arith.index_cast %intptr_69 : index to i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%32, %33, %35, %33) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview_70 = memref.subview %memref_30[0, 0] [%8, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_71 = memref.subview %memref_38[0, 0] [%10, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_72 = memref.subview %memref_22[0, 0] [%8, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_73 = memref.extract_aligned_pointer_as_index %subview_70 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %36 = arith.index_cast %intptr_73 : index to i64
    %base_buffer_74, %offset_75, %sizes_76:2, %strides_77:2 = memref.extract_strided_metadata %subview_70 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %37 = arith.index_cast %offset_75 : index to i64
    %38 = arith.muli %37, %c8_i64 : i64
    %39 = arith.addi %36, %38 : i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %intptr_78 = memref.extract_aligned_pointer_as_index %subview_71 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %41 = arith.index_cast %intptr_78 : index to i64
    %base_buffer_79, %offset_80, %sizes_81:2, %strides_82:2 = memref.extract_strided_metadata %subview_71 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %42 = arith.index_cast %offset_80 : index to i64
    %43 = arith.muli %42, %c8_i64 : i64
    %44 = arith.addi %41, %43 : i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_83 = memref.extract_aligned_pointer_as_index %subview_72 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %46 = arith.index_cast %intptr_83 : index to i64
    %base_buffer_84, %offset_85, %sizes_86:2, %strides_87:2 = memref.extract_strided_metadata %subview_72 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %47 = arith.index_cast %offset_85 : index to i64
    %48 = arith.muli %47, %c8_i64 : i64
    %49 = arith.addi %46, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg1, %arg3, %arg4, %cst, %40, %arg4, %45, %arg3, %cst_51, %50, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %dim_88 = memref.dim %memref_46, %c0_50 : memref<?x?xf64>
    %51 = arith.index_cast %dim_88 : index to i32
    %dim_89 = memref.dim %memref_46, %c1_49 : memref<?x?xf64>
    %52 = arith.index_cast %dim_89 : index to i32
    %intptr_90 = memref.extract_aligned_pointer_as_index %memref_46 : memref<?x?xf64> -> index
    %53 = arith.index_cast %intptr_90 : index to i64
    %54 = llvm.inttoptr %53 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%51, %52, %54, %52) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview_91 = memref.subview %memref_46[0, 0] [%12, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_92 = memref.extract_aligned_pointer_as_index %subview_55 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %55 = arith.index_cast %intptr_92 : index to i64
    %base_buffer_93, %offset_94, %sizes_95:2, %strides_96:2 = memref.extract_strided_metadata %subview_55 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %56 = arith.index_cast %offset_94 : index to i64
    %57 = arith.muli %56, %c8_i64 : i64
    %58 = arith.addi %55, %57 : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr
    %intptr_97 = memref.extract_aligned_pointer_as_index %subview_72 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %60 = arith.index_cast %intptr_97 : index to i64
    %base_buffer_98, %offset_99, %sizes_100:2, %strides_101:2 = memref.extract_strided_metadata %subview_72 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %61 = arith.index_cast %offset_99 : index to i64
    %62 = arith.muli %61, %c8_i64 : i64
    %63 = arith.addi %60, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %intptr_102 = memref.extract_aligned_pointer_as_index %subview_91 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %65 = arith.index_cast %intptr_102 : index to i64
    %base_buffer_103, %offset_104, %sizes_105:2, %strides_106:2 = memref.extract_strided_metadata %subview_91 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %66 = arith.index_cast %offset_104 : index to i64
    %67 = arith.muli %66, %c8_i64 : i64
    %68 = arith.addi %65, %67 : i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg3, %arg1, %cst, %59, %arg1, %64, %arg3, %cst_51, %69, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %70 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %71 = gpu.memcpy async [%70] %arg5, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %72 = gpu.dealloc async [%71] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_107 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_107) : (i32) -> ()
    %73 = gpu.memcpy async [%72] %arg6, %memref_6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_108 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_108) : (i32) -> ()
    %74 = gpu.dealloc async [%73] %memref_6 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_109 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_109) : (i32) -> ()
    %75 = gpu.memcpy async [%74] %arg7, %memref_14 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_110 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_110) : (i32) -> ()
    %76 = gpu.dealloc async [%75] %memref_14 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_111 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_111) : (i32) -> ()
    %77 = gpu.memcpy async [%76] %arg8, %memref_22 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_112 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_112) : (i32) -> ()
    %78 = gpu.dealloc async [%77] %memref_22 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_113 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_113) : (i32) -> ()
    %79 = gpu.memcpy async [%78] %arg9, %memref_30 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_114 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_114) : (i32) -> ()
    %80 = gpu.dealloc async [%79] %memref_30 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_115 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_115) : (i32) -> ()
    %81 = gpu.memcpy async [%80] %arg10, %memref_38 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_116 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_116) : (i32) -> ()
    %82 = gpu.dealloc async [%81] %memref_38 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_117 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_117) : (i32) -> ()
    %83 = gpu.memcpy async [%82] %arg11, %memref_46 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_118 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_118) : (i32) -> ()
    %84 = gpu.dealloc async [%83] %memref_46 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%84]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

