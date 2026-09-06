module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %4 = arith.index_cast %arg1 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg4, %c0 : memref<?xf64>
    %5 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %7 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %8 = arith.muli %7, %c8_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%5, %10) : (i32, !llvm.ptr) -> ()
    %11 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %12 = arith.index_cast %arg0 : i32 to index
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg5, %c0_0 : memref<?xf64>
    %13 = arith.index_cast %dim_1 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %14 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %15 = arith.index_cast %offset_4 : index to i64
    %c8_i64_7 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64_7 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%13, %18) : (i32, !llvm.ptr) -> ()
    %19 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%12, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_8 = tensor.extract_slice %1[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_9 = tensor.extract_slice %19[0] [%12] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_10 = arith.constant 1.000000e+00 : f64
    %20 = arith.index_cast %12 : index to i32
    %21 = arith.index_cast %4 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_11 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %22 = arith.index_cast %intptr_11 : index to i64
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
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
    %intptr_18 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %33 = arith.index_cast %intptr_18 : index to i64
    %base_buffer_19, %offset_20, %sizes_21, %strides_22 = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
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
    %subview = memref.subview %arg5[0] [%12] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %49 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %50 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    %51 = bufferization.to_memref %50 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %51, %arg5 : memref<?xf64> to memref<?xf64>
    %extracted_slice_32 = tensor.extract_slice %0[0, 0] [%12, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_33 = tensor.extract_slice %11[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_34 = arith.constant 1.000000e+00 : f64
    %52 = arith.index_cast %12 : index to i32
    %53 = arith.index_cast %4 : index to i32
    %intptr_35 = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
    %54 = arith.index_cast %intptr_35 : index to i64
    %base_buffer_36, %offset_37, %sizes_38:2, %strides_39:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %55 = arith.index_cast %offset_37 : index to i64
    %c0_i64_40 = arith.constant 0 : i64
    %56 = arith.index_cast %strides_39#0 : index to i64
    %57 = arith.muli %c0_i64_40, %56 : i64
    %58 = arith.addi %55, %57 : i64
    %c0_i64_41 = arith.constant 0 : i64
    %59 = arith.index_cast %strides_39#1 : index to i64
    %60 = arith.muli %c0_i64_41, %59 : i64
    %61 = arith.addi %58, %60 : i64
    %c8_i64_42 = arith.constant 8 : i64
    %62 = arith.muli %61, %c8_i64_42 : i64
    %63 = arith.addi %54, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %intptr_43 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %65 = arith.index_cast %intptr_43 : index to i64
    %base_buffer_44, %offset_45, %sizes_46, %strides_47 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %66 = arith.index_cast %offset_45 : index to i64
    %c8_i64_48 = arith.constant 8 : i64
    %67 = arith.muli %66, %c8_i64_48 : i64
    %68 = arith.addi %65, %67 : i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    %intptr_49 = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %70 = arith.index_cast %intptr_49 : index to i64
    %base_buffer_50, %offset_51, %sizes_52, %strides_53 = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %71 = arith.index_cast %offset_51 : index to i64
    %c0_i64_54 = arith.constant 0 : i64
    %72 = arith.index_cast %strides_53 : index to i64
    %73 = arith.muli %c0_i64_54, %72 : i64
    %74 = arith.addi %71, %73 : i64
    %c8_i64_55 = arith.constant 8 : i64
    %75 = arith.muli %74, %c8_i64_55 : i64
    %76 = arith.addi %70, %75 : i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%52, %53, %cst_34, %64, %53, %69, %cst_34, %77) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_56 = memref.subview %arg4[0] [%4] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %78 = bufferization.to_tensor %subview_56 restrict writable : memref<?xf64, strided<[1]>>
    %79 = bufferization.to_tensor %arg4 restrict writable : memref<?xf64>
    %80 = bufferization.to_memref %79 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %80, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

