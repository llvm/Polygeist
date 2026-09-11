module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg6, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg6, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg7, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg7, %c1_3 : memref<?x?xf64>
    %memref_5, %asyncToken_6 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    %2 = gpu.memcpy async [%asyncToken_6] %memref_5, %arg7 : memref<?x?xf64>, memref<?x?xf64>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg8, %c0_7 : memref<?x?xf64>
    %c1_9 = arith.constant 1 : index
    %dim_10 = memref.dim %arg8, %c1_9 : memref<?x?xf64>
    %memref_11, %asyncToken_12 = gpu.alloc async [%2] (%dim_8, %dim_10) : memref<?x?xf64>
    %3 = gpu.memcpy async [%asyncToken_12] %memref_11, %arg8 : memref<?x?xf64>, memref<?x?xf64>
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %arg9, %c0_13 : memref<?x?xf64>
    %c1_15 = arith.constant 1 : index
    %dim_16 = memref.dim %arg9, %c1_15 : memref<?x?xf64>
    %memref_17, %asyncToken_18 = gpu.alloc async [%3] (%dim_14, %dim_16) : memref<?x?xf64>
    %4 = gpu.memcpy async [%asyncToken_18] %memref_17, %arg9 : memref<?x?xf64>, memref<?x?xf64>
    %c0_19 = arith.constant 0 : index
    %dim_20 = memref.dim %arg10, %c0_19 : memref<?x?xf64>
    %c1_21 = arith.constant 1 : index
    %dim_22 = memref.dim %arg10, %c1_21 : memref<?x?xf64>
    %memref_23, %asyncToken_24 = gpu.alloc async [%4] (%dim_20, %dim_22) : memref<?x?xf64>
    %5 = gpu.memcpy async [%asyncToken_24] %memref_23, %arg10 : memref<?x?xf64>, memref<?x?xf64>
    gpu.wait [%5]
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %c1_25 = arith.constant 1 : index
    %c0_26 = arith.constant 0 : index
    %6 = arith.index_cast %arg2 : i32 to index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg0 : i32 to index
    %dim_27 = memref.dim %memref, %c0_26 : memref<?x?xf64>
    %10 = arith.index_cast %dim_27 : index to i32
    %dim_28 = memref.dim %memref, %c1_25 : memref<?x?xf64>
    %11 = arith.index_cast %dim_28 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %12 = arith.index_cast %intptr : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%10, %11, %13, %11) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview = memref.subview %memref_5[0, 0] [%9, %6] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_29 = memref.subview %memref_11[0, 0] [%6, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_30 = memref.subview %memref[0, 0] [%9, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_31 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %14 = arith.index_cast %intptr_31 : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %15 = arith.index_cast %offset : index to i64
    %16 = arith.muli %15, %c8_i64 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %intptr_32 = memref.extract_aligned_pointer_as_index %subview_29 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %19 = arith.index_cast %intptr_32 : index to i64
    %base_buffer_33, %offset_34, %sizes_35:2, %strides_36:2 = memref.extract_strided_metadata %subview_29 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %20 = arith.index_cast %offset_34 : index to i64
    %21 = arith.muli %20, %c8_i64 : i64
    %22 = arith.addi %19, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %intptr_37 = memref.extract_aligned_pointer_as_index %subview_30 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %24 = arith.index_cast %intptr_37 : index to i64
    %base_buffer_38, %offset_39, %sizes_40:2, %strides_41:2 = memref.extract_strided_metadata %subview_30 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %25 = arith.index_cast %offset_39 : index to i64
    %26 = arith.muli %25, %c8_i64 : i64
    %27 = arith.addi %24, %26 : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%arg0, %arg1, %arg2, %arg4, %18, %arg2, %23, %arg1, %cst, %28, %arg1) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %subview_42 = memref.subview %memref_17[0, 0] [%8, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_43 = memref.subview %memref_23[0, 0] [%9, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_44 = memref.extract_aligned_pointer_as_index %subview_30 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %29 = arith.index_cast %intptr_44 : index to i64
    %base_buffer_45, %offset_46, %sizes_47:2, %strides_48:2 = memref.extract_strided_metadata %subview_30 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %30 = arith.index_cast %offset_46 : index to i64
    %31 = arith.muli %30, %c8_i64 : i64
    %32 = arith.addi %29, %31 : i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %intptr_49 = memref.extract_aligned_pointer_as_index %subview_42 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %34 = arith.index_cast %intptr_49 : index to i64
    %base_buffer_50, %offset_51, %sizes_52:2, %strides_53:2 = memref.extract_strided_metadata %subview_42 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %35 = arith.index_cast %offset_51 : index to i64
    %36 = arith.muli %35, %c8_i64 : i64
    %37 = arith.addi %34, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %intptr_54 = memref.extract_aligned_pointer_as_index %subview_43 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %39 = arith.index_cast %intptr_54 : index to i64
    %base_buffer_55, %offset_56, %sizes_57:2, %strides_58:2 = memref.extract_strided_metadata %subview_43 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %40 = arith.index_cast %offset_56 : index to i64
    %41 = arith.muli %40, %c8_i64 : i64
    %42 = arith.addi %39, %41 : i64
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemm(%arg0, %arg3, %arg1, %cst, %33, %arg1, %38, %arg3, %arg5, %43, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %44 = gpu.wait async
    %45 = gpu.memcpy async [%44] %arg6, %memref : memref<?x?xf64>, memref<?x?xf64>
    %46 = gpu.dealloc async [%45] %memref : memref<?x?xf64>
    %47 = gpu.memcpy async [%46] %arg7, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %48 = gpu.dealloc async [%47] %memref_5 : memref<?x?xf64>
    %49 = gpu.memcpy async [%48] %arg8, %memref_11 : memref<?x?xf64>, memref<?x?xf64>
    %50 = gpu.dealloc async [%49] %memref_11 : memref<?x?xf64>
    %51 = gpu.memcpy async [%50] %arg9, %memref_17 : memref<?x?xf64>, memref<?x?xf64>
    %52 = gpu.dealloc async [%51] %memref_17 : memref<?x?xf64>
    %53 = gpu.memcpy async [%52] %arg10, %memref_23 : memref<?x?xf64>, memref<?x?xf64>
    %54 = gpu.dealloc async [%53] %memref_23 : memref<?x?xf64>
    gpu.wait [%54]
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

