module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_mvt(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg1 restrict : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %4[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst = arith.constant 1.000000e+00 : f64
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.index_cast %5 : index to i32
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %4, %c1 : tensor<?x?xf64>
    %8 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %9 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %10 = arith.index_cast %offset : index to i64
    %c0_i64 = arith.constant 0 : i64
    %11 = arith.index_cast %strides#0 : index to i64
    %12 = arith.muli %c0_i64, %11 : i64
    %13 = arith.addi %10, %12 : i64
    %c0_i64_2 = arith.constant 0 : i64
    %14 = arith.index_cast %strides#1 : index to i64
    %15 = arith.muli %c0_i64_2, %14 : i64
    %16 = arith.addi %13, %15 : i64
    %c8_i64 = arith.constant 8 : i64
    %17 = arith.muli %16, %c8_i64 : i64
    %18 = arith.addi %9, %17 : i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %20 = arith.index_cast %intptr_3 : index to i64
    %base_buffer_4, %offset_5, %sizes_6, %strides_7 = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
    %21 = arith.index_cast %offset_5 : index to i64
    %c0_i64_8 = arith.constant 0 : i64
    %22 = arith.index_cast %strides_7 : index to i64
    %23 = arith.muli %c0_i64_8, %22 : i64
    %24 = arith.addi %21, %23 : i64
    %c8_i64_9 = arith.constant 8 : i64
    %25 = arith.muli %24, %c8_i64_9 : i64
    %26 = arith.addi %20, %25 : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    %subview = memref.subview %arg1[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_10 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %28 = arith.index_cast %intptr_10 : index to i64
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %29 = arith.index_cast %offset_12 : index to i64
    %c8_i64_15 = arith.constant 8 : i64
    %30 = arith.muli %29, %c8_i64_15 : i64
    %31 = arith.addi %28, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%6, %7, %cst, %19, %8, %27, %cst, %32) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %33 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %34 = bufferization.to_tensor %arg1 restrict writable : memref<?xf64>
    %35 = bufferization.to_memref %34 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %35, %arg1 : memref<?xf64> to memref<?xf64>
    %extracted_slice_16 = tensor.extract_slice %4[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_17 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_18 = tensor.extract_slice %1[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_19 = arith.constant 1.000000e+00 : f64
    %36 = arith.index_cast %5 : index to i32
    %37 = arith.index_cast %5 : index to i32
    %c1_20 = arith.constant 1 : index
    %dim_21 = tensor.dim %4, %c1_20 : tensor<?x?xf64>
    %38 = arith.index_cast %dim_21 : index to i32
    %intptr_22 = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %39 = arith.index_cast %intptr_22 : index to i64
    %base_buffer_23, %offset_24, %sizes_25:2, %strides_26:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %40 = arith.index_cast %offset_24 : index to i64
    %c0_i64_27 = arith.constant 0 : i64
    %41 = arith.index_cast %strides_26#0 : index to i64
    %42 = arith.muli %c0_i64_27, %41 : i64
    %43 = arith.addi %40, %42 : i64
    %c0_i64_28 = arith.constant 0 : i64
    %44 = arith.index_cast %strides_26#1 : index to i64
    %45 = arith.muli %c0_i64_28, %44 : i64
    %46 = arith.addi %43, %45 : i64
    %c8_i64_29 = arith.constant 8 : i64
    %47 = arith.muli %46, %c8_i64_29 : i64
    %48 = arith.addi %39, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %intptr_30 = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %50 = arith.index_cast %intptr_30 : index to i64
    %base_buffer_31, %offset_32, %sizes_33, %strides_34 = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %51 = arith.index_cast %offset_32 : index to i64
    %c0_i64_35 = arith.constant 0 : i64
    %52 = arith.index_cast %strides_34 : index to i64
    %53 = arith.muli %c0_i64_35, %52 : i64
    %54 = arith.addi %51, %53 : i64
    %c8_i64_36 = arith.constant 8 : i64
    %55 = arith.muli %54, %c8_i64_36 : i64
    %56 = arith.addi %50, %55 : i64
    %57 = llvm.inttoptr %56 : i64 to !llvm.ptr
    %subview_37 = memref.subview %arg2[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_38 = memref.extract_aligned_pointer_as_index %subview_37 : memref<?xf64, strided<[1]>> -> index
    %58 = arith.index_cast %intptr_38 : index to i64
    %base_buffer_39, %offset_40, %sizes_41, %strides_42 = memref.extract_strided_metadata %subview_37 : memref<?xf64, strided<[1]>> -> memref<f64>, index, index, index
    %59 = arith.index_cast %offset_40 : index to i64
    %c8_i64_43 = arith.constant 8 : i64
    %60 = arith.muli %59, %c8_i64_43 : i64
    %61 = arith.addi %58, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%36, %37, %cst_19, %49, %38, %57, %cst_19, %62) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %63 = bufferization.to_tensor %subview_37 restrict writable : memref<?xf64, strided<[1]>>
    %64 = bufferization.to_tensor %arg2 restrict writable : memref<?xf64>
    %65 = bufferization.to_memref %64 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    memref.copy %65, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

