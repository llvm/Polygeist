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
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_11 = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?xf64> -> index
    %22 = arith.index_cast %intptr_11 : index to i64
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %23 = arith.index_cast %offset_13 : index to i64
    %c0_i64 = arith.constant 0 : i64
    %24 = arith.index_cast %strides_15#0 : index to i64
    %25 = arith.muli %c0_i64, %24 : i64
    %26 = arith.addi %23, %25 : i64
    %c0_i64_16 = arith.constant 0 : i64
    %27 = arith.index_cast %strides_15#1 : index to i64
    %28 = arith.muli %c0_i64_16, %27 : i64
    %29 = arith.addi %26, %28 : i64
    %c8_i64_17 = arith.constant 8 : i64
    %30 = arith.muli %29, %c8_i64_17 : i64
    %31 = arith.addi %22, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %33 = arith.index_cast %intptr_18 : index to i64
    %base_buffer_19, %offset_20, %sizes_21, %strides_22 = memref.extract_strided_metadata %arg6 : memref<?xf64> -> memref<f64>, index, index, index
    %34 = arith.index_cast %offset_20 : index to i64
    %c0_i64_23 = arith.constant 0 : i64
    %35 = arith.index_cast %strides_22 : index to i64
    %36 = arith.muli %c0_i64_23, %35 : i64
    %37 = arith.addi %34, %36 : i64
    %c8_i64_24 = arith.constant 8 : i64
    %38 = arith.muli %37, %c8_i64_24 : i64
    %39 = arith.addi %33, %38 : i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %intptr_25 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %41 = arith.index_cast %intptr_25 : index to i64
    %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %42 = arith.index_cast %offset_27 : index to i64
    %c0_i64_30 = arith.constant 0 : i64
    %43 = arith.index_cast %strides_29 : index to i64
    %44 = arith.muli %c0_i64_30, %43 : i64
    %45 = arith.addi %42, %44 : i64
    %c8_i64_31 = arith.constant 8 : i64
    %46 = arith.muli %45, %c8_i64_31 : i64
    %47 = arith.addi %41, %46 : i64
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%20, %21, %cst_10, %32, %21, %40, %cst_10, %48) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview = memref.subview %arg5[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %49 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %50 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    %51 = bufferization.to_memref %50 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %51, %arg5 : memref<?xf64> to memref<?xf64>
    %extracted_slice_32 = tensor.extract_slice %1[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_34 = tensor.extract_slice %19[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_35 = arith.constant 1.000000e+00 : f64
    %52 = arith.index_cast %5 : index to i32
    %53 = arith.index_cast %5 : index to i32
    %intptr_36 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %54 = arith.index_cast %intptr_36 : index to i64
    %base_buffer_37, %offset_38, %sizes_39:2, %strides_40:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %55 = arith.index_cast %offset_38 : index to i64
    %c0_i64_41 = arith.constant 0 : i64
    %56 = arith.index_cast %strides_40#0 : index to i64
    %57 = arith.muli %c0_i64_41, %56 : i64
    %58 = arith.addi %55, %57 : i64
    %c0_i64_42 = arith.constant 0 : i64
    %59 = arith.index_cast %strides_40#1 : index to i64
    %60 = arith.muli %c0_i64_42, %59 : i64
    %61 = arith.addi %58, %60 : i64
    %c8_i64_43 = arith.constant 8 : i64
    %62 = arith.muli %61, %c8_i64_43 : i64
    %63 = arith.addi %54, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %intptr_44 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %65 = arith.index_cast %intptr_44 : index to i64
    %base_buffer_45, %offset_46, %sizes_47, %strides_48 = memref.extract_strided_metadata %arg6 : memref<?xf64> -> memref<f64>, index, index, index
    %66 = arith.index_cast %offset_46 : index to i64
    %c0_i64_49 = arith.constant 0 : i64
    %67 = arith.index_cast %strides_48 : index to i64
    %68 = arith.muli %c0_i64_49, %67 : i64
    %69 = arith.addi %66, %68 : i64
    %c8_i64_50 = arith.constant 8 : i64
    %70 = arith.muli %69, %c8_i64_50 : i64
    %71 = arith.addi %65, %70 : i64
    %72 = llvm.inttoptr %71 : i64 to !llvm.ptr
    %intptr_51 = memref.extract_aligned_pointer_as_index %arg7 : memref<?xf64> -> index
    %73 = arith.index_cast %intptr_51 : index to i64
    %base_buffer_52, %offset_53, %sizes_54, %strides_55 = memref.extract_strided_metadata %arg7 : memref<?xf64> -> memref<f64>, index, index, index
    %74 = arith.index_cast %offset_53 : index to i64
    %c0_i64_56 = arith.constant 0 : i64
    %75 = arith.index_cast %strides_55 : index to i64
    %76 = arith.muli %c0_i64_56, %75 : i64
    %77 = arith.addi %74, %76 : i64
    %c8_i64_57 = arith.constant 8 : i64
    %78 = arith.muli %77, %c8_i64_57 : i64
    %79 = arith.addi %73, %78 : i64
    %80 = llvm.inttoptr %79 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%52, %53, %cst_35, %64, %53, %72, %cst_35, %80) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_58 = memref.subview %arg7[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %81 = bufferization.to_tensor %subview_58 restrict writable : memref<?xf64, strided<[1]>>
    %82 = bufferization.to_tensor %arg7 restrict writable : memref<?xf64>
    %c0_59 = arith.constant 0 : index
    %dim_60 = memref.dim %arg7, %c0_59 : memref<?xf64>
    %83 = arith.index_cast %dim_60 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_61 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %84 = arith.index_cast %intptr_61 : index to i64
    %base_buffer_62, %offset_63, %sizes_64, %strides_65 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %85 = arith.index_cast %offset_63 : index to i64
    %c8_i64_66 = arith.constant 8 : i64
    %86 = arith.muli %85, %c8_i64_66 : i64
    %87 = arith.addi %84, %86 : i64
    %88 = llvm.inttoptr %87 : i64 to !llvm.ptr
    %intptr_67 = memref.extract_aligned_pointer_as_index %arg7 : memref<?xf64> -> index
    %89 = arith.index_cast %intptr_67 : index to i64
    %base_buffer_68, %offset_69, %sizes_70, %strides_71 = memref.extract_strided_metadata %arg7 : memref<?xf64> -> memref<f64>, index, index, index
    %90 = arith.index_cast %offset_69 : index to i64
    %c8_i64_72 = arith.constant 8 : i64
    %91 = arith.muli %90, %c8_i64_72 : i64
    %92 = arith.addi %89, %91 : i64
    %93 = llvm.inttoptr %92 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_daxpby(%83, %arg1, %88, %arg2, %93) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    %94 = bufferization.to_tensor %arg7 restrict writable : memref<?xf64>
    %95 = bufferization.to_memref %94 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %95, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

