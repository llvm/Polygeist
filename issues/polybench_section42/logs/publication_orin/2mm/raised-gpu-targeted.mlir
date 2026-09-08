module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -1092367658908009857 : i64, polygeist.pipeline_scope_coalesced} {
    %c-1092367658908009857_i64 = arith.constant -1092367658908009857 : i64
    call @polygeist_gpu_region_timing_begin(%c-1092367658908009857_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg6, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg6, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg7, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg7, %c1_3 : memref<?x?xf64>
    %c2_i32_5 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_5) : (i32) -> ()
    %memref_6, %asyncToken_7 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_8 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_8) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_7] %memref_6, %arg7 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg8, %c0_9 : memref<?x?xf64>
    %c1_11 = arith.constant 1 : index
    %dim_12 = memref.dim %arg8, %c1_11 : memref<?x?xf64>
    %c2_i32_13 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_13) : (i32) -> ()
    %memref_14, %asyncToken_15 = gpu.alloc async [%2] (%dim_10, %dim_12) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_16 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_16) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_15] %memref_14, %arg8 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_17 = arith.constant 0 : index
    %dim_18 = memref.dim %arg9, %c0_17 : memref<?x?xf64>
    %c1_19 = arith.constant 1 : index
    %dim_20 = memref.dim %arg9, %c1_19 : memref<?x?xf64>
    %c2_i32_21 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_21) : (i32) -> ()
    %memref_22, %asyncToken_23 = gpu.alloc async [%3] (%dim_18, %dim_20) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_24 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_24) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_23] %memref_22, %arg9 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg10, %c0_25 : memref<?x?xf64>
    %c1_27 = arith.constant 1 : index
    %dim_28 = memref.dim %arg10, %c1_27 : memref<?x?xf64>
    %c2_i32_29 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_29) : (i32) -> ()
    %memref_30, %asyncToken_31 = gpu.alloc async [%4] (%dim_26, %dim_28) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_32) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_31] %memref_30, %arg10 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%5]
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %c1_33 = arith.constant 1 : index
    %c0_34 = arith.constant 0 : index
    %6 = arith.index_cast %arg2 : i32 to index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg0 : i32 to index
    %dim_35 = memref.dim %memref, %c0_34 : memref<?x?xf64>
    %10 = arith.index_cast %dim_35 : index to i32
    %dim_36 = memref.dim %memref, %c1_33 : memref<?x?xf64>
    %11 = arith.index_cast %dim_36 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %12 = arith.index_cast %intptr : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%10, %11, %13, %11) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview = memref.subview %memref_6[0, 0] [%9, %6] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_37 = memref.subview %memref_14[0, 0] [%6, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_38 = memref.subview %memref[0, 0] [%9, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_39 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %14 = arith.index_cast %intptr_39 : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %15 = arith.index_cast %offset : index to i64
    %16 = arith.muli %15, %c8_i64 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %intptr_40 = memref.extract_aligned_pointer_as_index %subview_37 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %19 = arith.index_cast %intptr_40 : index to i64
    %base_buffer_41, %offset_42, %sizes_43:2, %strides_44:2 = memref.extract_strided_metadata %subview_37 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %20 = arith.index_cast %offset_42 : index to i64
    %21 = arith.muli %20, %c8_i64 : i64
    %22 = arith.addi %19, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %intptr_45 = memref.extract_aligned_pointer_as_index %subview_38 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %24 = arith.index_cast %intptr_45 : index to i64
    %base_buffer_46, %offset_47, %sizes_48:2, %strides_49:2 = memref.extract_strided_metadata %subview_38 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %25 = arith.index_cast %offset_47 : index to i64
    %26 = arith.muli %25, %c8_i64 : i64
    %27 = arith.addi %24, %26 : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg1, %arg2, %arg4, %18, %arg2, %23, %arg1, %cst, %28, %arg1) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %subview_50 = memref.subview %memref_22[0, 0] [%8, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_51 = memref.subview %memref_30[0, 0] [%9, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_52 = memref.extract_aligned_pointer_as_index %subview_38 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %29 = arith.index_cast %intptr_52 : index to i64
    %base_buffer_53, %offset_54, %sizes_55:2, %strides_56:2 = memref.extract_strided_metadata %subview_38 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %30 = arith.index_cast %offset_54 : index to i64
    %31 = arith.muli %30, %c8_i64 : i64
    %32 = arith.addi %29, %31 : i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %intptr_57 = memref.extract_aligned_pointer_as_index %subview_50 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %34 = arith.index_cast %intptr_57 : index to i64
    %base_buffer_58, %offset_59, %sizes_60:2, %strides_61:2 = memref.extract_strided_metadata %subview_50 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %35 = arith.index_cast %offset_59 : index to i64
    %36 = arith.muli %35, %c8_i64 : i64
    %37 = arith.addi %34, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %intptr_62 = memref.extract_aligned_pointer_as_index %subview_51 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %39 = arith.index_cast %intptr_62 : index to i64
    %base_buffer_63, %offset_64, %sizes_65:2, %strides_66:2 = memref.extract_strided_metadata %subview_51 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %40 = arith.index_cast %offset_64 : index to i64
    %41 = arith.muli %40, %c8_i64 : i64
    %42 = arith.addi %39, %41 : i64
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg3, %arg1, %cst, %33, %arg1, %38, %arg3, %arg5, %43, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %44 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %45 = gpu.memcpy async [%44] %arg6, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %46 = gpu.dealloc async [%45] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_67 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_67) : (i32) -> ()
    %47 = gpu.memcpy async [%46] %arg7, %memref_6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_68 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_68) : (i32) -> ()
    %48 = gpu.dealloc async [%47] %memref_6 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_69 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_69) : (i32) -> ()
    %49 = gpu.memcpy async [%48] %arg8, %memref_14 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_70 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_70) : (i32) -> ()
    %50 = gpu.dealloc async [%49] %memref_14 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_71 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_71) : (i32) -> ()
    %51 = gpu.memcpy async [%50] %arg9, %memref_22 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_72 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_72) : (i32) -> ()
    %52 = gpu.dealloc async [%51] %memref_22 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_73 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_73) : (i32) -> ()
    %53 = gpu.memcpy async [%52] %arg10, %memref_30 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_74 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_74) : (i32) -> ()
    %54 = gpu.dealloc async [%53] %memref_30 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%54]
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

