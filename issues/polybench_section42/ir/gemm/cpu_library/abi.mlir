module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg7 restrict : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %1[0, 0] [%5, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%4, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0] [%5, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %subview = memref.subview %arg6[0, 0] [%5, %4] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_2 = memref.subview %arg7[0, 0] [%4, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_3 = memref.subview %arg5[0, 0] [%5, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %subview, %c0 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %6 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_4 = memref.dim %subview, %c1 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %7 = arith.index_cast %dim_4 : index to i32
    %c1_5 = arith.constant 1 : index
    %dim_6 = memref.dim %subview_2, %c1_5 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %8 = arith.index_cast %dim_6 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %9 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %10 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %11 = arith.muli %10, %c8_i64 : i64
    %12 = arith.addi %9, %11 : i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %intptr_7 = memref.extract_aligned_pointer_as_index %subview_2 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %14 = arith.index_cast %intptr_7 : index to i64
    %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %subview_2 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %15 = arith.index_cast %offset_9 : index to i64
    %c8_i64_12 = arith.constant 8 : i64
    %16 = arith.muli %15, %c8_i64_12 : i64
    %17 = arith.addi %14, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %intptr_13 = memref.extract_aligned_pointer_as_index %subview_3 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %19 = arith.index_cast %intptr_13 : index to i64
    %base_buffer_14, %offset_15, %sizes_16:2, %strides_17:2 = memref.extract_strided_metadata %subview_3 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %20 = arith.index_cast %offset_15 : index to i64
    %c8_i64_18 = arith.constant 8 : i64
    %21 = arith.muli %20, %c8_i64_18 : i64
    %22 = arith.addi %19, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%6, %8, %7, %arg3, %13, %7, %18, %8, %arg4, %23, %8) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %24 = bufferization.to_tensor %subview_3 restrict writable : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %25 = bufferization.to_tensor %arg5 restrict writable : memref<?x?xf64>
    %26 = bufferization.to_memref %25 : memref<?x?xf64>
    memref.copy %26, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
}

