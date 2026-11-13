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
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %9 = arith.index_cast %offset : index to i64
    %c0_i64 = arith.constant 0 : i64
    %10 = arith.index_cast %strides#0 : index to i64
    %11 = arith.muli %c0_i64, %10 : i64
    %12 = arith.addi %9, %11 : i64
    %c0_i64_2 = arith.constant 0 : i64
    %13 = arith.index_cast %strides#1 : index to i64
    %14 = arith.muli %c0_i64_2, %13 : i64
    %15 = arith.addi %12, %14 : i64
    %c8_i64 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64 : i64
    %17 = arith.addi %8, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %19 = arith.index_cast %intptr_3 : index to i64
    %base_buffer_4, %offset_5, %sizes_6, %strides_7 = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
    %20 = arith.index_cast %offset_5 : index to i64
    %c0_i64_8 = arith.constant 0 : i64
    %21 = arith.index_cast %strides_7 : index to i64
    %22 = arith.muli %c0_i64_8, %21 : i64
    %23 = arith.addi %20, %22 : i64
    %c8_i64_9 = arith.constant 8 : i64
    %24 = arith.muli %23, %c8_i64_9 : i64
    %25 = arith.addi %19, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %intptr_10 = memref.extract_aligned_pointer_as_index %arg1 : memref<?xf64> -> index
    %27 = arith.index_cast %intptr_10 : index to i64
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %arg1 : memref<?xf64> -> memref<f64>, index, index, index
    %28 = arith.index_cast %offset_12 : index to i64
    %c0_i64_15 = arith.constant 0 : i64
    %29 = arith.index_cast %strides_14 : index to i64
    %30 = arith.muli %c0_i64_15, %29 : i64
    %31 = arith.addi %28, %30 : i64
    %c8_i64_16 = arith.constant 8 : i64
    %32 = arith.muli %31, %c8_i64_16 : i64
    %33 = arith.addi %27, %32 : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%6, %7, %cst, %18, %7, %26, %cst, %34) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview = memref.subview %arg1[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %35 = bufferization.to_tensor %subview restrict writable : memref<?xf64, strided<[1]>>
    %36 = bufferization.to_tensor %arg1 restrict writable : memref<?xf64>
    %37 = bufferization.to_memref %36 : memref<?xf64>
    memref.copy %37, %arg1 : memref<?xf64> to memref<?xf64>
    %extracted_slice_17 = tensor.extract_slice %4[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_18 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_19 = tensor.extract_slice %1[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %cst_20 = arith.constant 1.000000e+00 : f64
    %38 = arith.index_cast %5 : index to i32
    %39 = arith.index_cast %5 : index to i32
    %intptr_21 = memref.extract_aligned_pointer_as_index %arg5 : memref<?x?xf64> -> index
    %40 = arith.index_cast %intptr_21 : index to i64
    %base_buffer_22, %offset_23, %sizes_24:2, %strides_25:2 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %41 = arith.index_cast %offset_23 : index to i64
    %c0_i64_26 = arith.constant 0 : i64
    %42 = arith.index_cast %strides_25#0 : index to i64
    %43 = arith.muli %c0_i64_26, %42 : i64
    %44 = arith.addi %41, %43 : i64
    %c0_i64_27 = arith.constant 0 : i64
    %45 = arith.index_cast %strides_25#1 : index to i64
    %46 = arith.muli %c0_i64_27, %45 : i64
    %47 = arith.addi %44, %46 : i64
    %c8_i64_28 = arith.constant 8 : i64
    %48 = arith.muli %47, %c8_i64_28 : i64
    %49 = arith.addi %40, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %intptr_29 = memref.extract_aligned_pointer_as_index %arg4 : memref<?xf64> -> index
    %51 = arith.index_cast %intptr_29 : index to i64
    %base_buffer_30, %offset_31, %sizes_32, %strides_33 = memref.extract_strided_metadata %arg4 : memref<?xf64> -> memref<f64>, index, index, index
    %52 = arith.index_cast %offset_31 : index to i64
    %c0_i64_34 = arith.constant 0 : i64
    %53 = arith.index_cast %strides_33 : index to i64
    %54 = arith.muli %c0_i64_34, %53 : i64
    %55 = arith.addi %52, %54 : i64
    %c8_i64_35 = arith.constant 8 : i64
    %56 = arith.muli %55, %c8_i64_35 : i64
    %57 = arith.addi %51, %56 : i64
    %58 = llvm.inttoptr %57 : i64 to !llvm.ptr
    %intptr_36 = memref.extract_aligned_pointer_as_index %arg2 : memref<?xf64> -> index
    %59 = arith.index_cast %intptr_36 : index to i64
    %base_buffer_37, %offset_38, %sizes_39, %strides_40 = memref.extract_strided_metadata %arg2 : memref<?xf64> -> memref<f64>, index, index, index
    %60 = arith.index_cast %offset_38 : index to i64
    %c0_i64_41 = arith.constant 0 : i64
    %61 = arith.index_cast %strides_40 : index to i64
    %62 = arith.muli %c0_i64_41, %61 : i64
    %63 = arith.addi %60, %62 : i64
    %c8_i64_42 = arith.constant 8 : i64
    %64 = arith.muli %63, %c8_i64_42 : i64
    %65 = arith.addi %59, %64 : i64
    %66 = llvm.inttoptr %65 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv_T(%38, %39, %cst_20, %50, %39, %58, %cst_20, %66) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_43 = memref.subview %arg2[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %67 = bufferization.to_tensor %subview_43 restrict writable : memref<?xf64, strided<[1]>>
    %68 = bufferization.to_tensor %arg2 restrict writable : memref<?xf64>
    %69 = bufferization.to_memref %68 : memref<?xf64>
    memref.copy %69, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
}

