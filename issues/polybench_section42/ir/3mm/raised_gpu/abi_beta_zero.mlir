module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %dim = memref.dim %arg5, %c0 : memref<?x?xf64>
    %5 = arith.index_cast %dim : index to i32
    %dim_1 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %6 = arith.index_cast %dim_1 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %7 = arith.index_cast %intptr : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%5, %6, %8, %6) : (i32, i32, !llvm.ptr, i32) -> ()
    %9 = bufferization.to_tensor %arg5 restrict writable : memref<?x?xf64>
    %subview = memref.subview %arg6[0, 0] [%4, %1] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_2 = memref.subview %arg7[0, 0] [%1, %0] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_3 = memref.subview %arg5[0, 0] [%4, %0] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_4 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %10 = arith.index_cast %intptr_4 : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %11 = arith.index_cast %offset : index to i64
    %12 = arith.muli %11, %c8_i64 : i64
    %13 = arith.addi %10, %12 : i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %intptr_5 = memref.extract_aligned_pointer_as_index %subview_2 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %15 = arith.index_cast %intptr_5 : index to i64
    %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %subview_2 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %16 = arith.index_cast %offset_7 : index to i64
    %17 = arith.muli %16, %c8_i64 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %intptr_10 = memref.extract_aligned_pointer_as_index %subview_3 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %20 = arith.index_cast %intptr_10 : index to i64
    %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %subview_3 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %21 = arith.index_cast %offset_12 : index to i64
    %22 = arith.muli %21, %c8_i64 : i64
    %23 = arith.addi %20, %22 : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg1, %arg2, %cst, %14, %arg2, %19, %arg1, %cst_0, %24, %arg1) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %25 = bufferization.to_tensor %subview_3 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %inserted_slice = tensor.insert_slice %25 into %9[0, 0] [%4, %0] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %26 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %26, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %dim_15 = memref.dim %arg8, %c0 : memref<?x?xf64>
    %27 = arith.index_cast %dim_15 : index to i32
    %dim_16 = memref.dim %arg8, %c1 : memref<?x?xf64>
    %28 = arith.index_cast %dim_16 : index to i32
    %intptr_17 = memref.extract_aligned_pointer_as_index %arg8 : memref<?x?xf64> -> index
    %29 = arith.index_cast %intptr_17 : index to i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%27, %28, %30, %28) : (i32, i32, !llvm.ptr, i32) -> ()
    %31 = bufferization.to_tensor %arg8 restrict writable : memref<?x?xf64>
    %subview_18 = memref.subview %arg9[0, 0] [%0, %2] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_19 = memref.subview %arg10[0, 0] [%2, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_20 = memref.subview %arg8[0, 0] [%0, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_21 = memref.extract_aligned_pointer_as_index %subview_18 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %32 = arith.index_cast %intptr_21 : index to i64
    %base_buffer_22, %offset_23, %sizes_24:2, %strides_25:2 = memref.extract_strided_metadata %subview_18 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %33 = arith.index_cast %offset_23 : index to i64
    %34 = arith.muli %33, %c8_i64 : i64
    %35 = arith.addi %32, %34 : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %intptr_26 = memref.extract_aligned_pointer_as_index %subview_19 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %37 = arith.index_cast %intptr_26 : index to i64
    %base_buffer_27, %offset_28, %sizes_29:2, %strides_30:2 = memref.extract_strided_metadata %subview_19 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %38 = arith.index_cast %offset_28 : index to i64
    %39 = arith.muli %38, %c8_i64 : i64
    %40 = arith.addi %37, %39 : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %intptr_31 = memref.extract_aligned_pointer_as_index %subview_20 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %42 = arith.index_cast %intptr_31 : index to i64
    %base_buffer_32, %offset_33, %sizes_34:2, %strides_35:2 = memref.extract_strided_metadata %subview_20 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %43 = arith.index_cast %offset_33 : index to i64
    %44 = arith.muli %43, %c8_i64 : i64
    %45 = arith.addi %42, %44 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg1, %arg3, %arg4, %cst, %36, %arg4, %41, %arg3, %cst_0, %46, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %47 = bufferization.to_tensor %subview_20 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %inserted_slice_36 = tensor.insert_slice %47 into %31[0, 0] [%0, %3] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %48 = bufferization.to_memref %inserted_slice_36 : memref<?x?xf64>
    memref.copy %48, %arg8 : memref<?x?xf64> to memref<?x?xf64>
    %dim_37 = memref.dim %arg11, %c0 : memref<?x?xf64>
    %49 = arith.index_cast %dim_37 : index to i32
    %dim_38 = memref.dim %arg11, %c1 : memref<?x?xf64>
    %50 = arith.index_cast %dim_38 : index to i32
    %intptr_39 = memref.extract_aligned_pointer_as_index %arg11 : memref<?x?xf64> -> index
    %51 = arith.index_cast %intptr_39 : index to i64
    %52 = llvm.inttoptr %51 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%49, %50, %52, %50) : (i32, i32, !llvm.ptr, i32) -> ()
    %53 = bufferization.to_tensor %arg11 restrict writable : memref<?x?xf64>
    %subview_40 = memref.subview %arg11[0, 0] [%4, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_41 = memref.extract_aligned_pointer_as_index %subview_3 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %54 = arith.index_cast %intptr_41 : index to i64
    %base_buffer_42, %offset_43, %sizes_44:2, %strides_45:2 = memref.extract_strided_metadata %subview_3 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %55 = arith.index_cast %offset_43 : index to i64
    %56 = arith.muli %55, %c8_i64 : i64
    %57 = arith.addi %54, %56 : i64
    %58 = llvm.inttoptr %57 : i64 to !llvm.ptr
    %intptr_46 = memref.extract_aligned_pointer_as_index %subview_20 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %59 = arith.index_cast %intptr_46 : index to i64
    %base_buffer_47, %offset_48, %sizes_49:2, %strides_50:2 = memref.extract_strided_metadata %subview_20 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %60 = arith.index_cast %offset_48 : index to i64
    %61 = arith.muli %60, %c8_i64 : i64
    %62 = arith.addi %59, %61 : i64
    %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
    %intptr_51 = memref.extract_aligned_pointer_as_index %subview_40 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %64 = arith.index_cast %intptr_51 : index to i64
    %base_buffer_52, %offset_53, %sizes_54:2, %strides_55:2 = memref.extract_strided_metadata %subview_40 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %65 = arith.index_cast %offset_53 : index to i64
    %66 = arith.muli %65, %c8_i64 : i64
    %67 = arith.addi %64, %66 : i64
    %68 = llvm.inttoptr %67 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg3, %arg1, %cst, %58, %arg1, %63, %arg3, %cst_0, %68, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %69 = bufferization.to_tensor %subview_40 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %inserted_slice_56 = tensor.insert_slice %69 into %53[0, 0] [%4, %3] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %70 = bufferization.to_memref %inserted_slice_56 : memref<?x?xf64>
    memref.copy %70, %arg11 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
}

