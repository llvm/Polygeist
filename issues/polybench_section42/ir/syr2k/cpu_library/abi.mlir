#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.subi %4, %c1 : index
    %6 = affine.apply #map(%5)
    %extracted_slice = tensor.extract_slice %1[0, 0] [%6, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%4, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %2[0, 0] [%6, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %1[0, 0] [%4, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %0[0, 0] [%4, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %subview = memref.subview %arg5[0, 0] [%6, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_4 = memref.subview %arg6[0, 0] [%4, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %subview, %c0 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %7 = arith.index_cast %dim : index to i32
    %c1_5 = arith.constant 1 : index
    %dim_6 = memref.dim %subview, %c1_5 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %8 = arith.index_cast %dim_6 : index to i32
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %base_buffer_7, %offset_8, %sizes_9:2, %strides_10:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %9 = arith.index_cast %strides#0 : index to i32
    %10 = arith.index_cast %strides_10#0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %offset_12 : index to i64
    %c8_i64 = arith.constant 8 : i64
    %13 = arith.muli %12, %c8_i64 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_15 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %16 = arith.index_cast %intptr_15 : index to i64
    %base_buffer_16, %offset_17, %sizes_18:2, %strides_19:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %17 = arith.index_cast %offset_17 : index to i64
    %c8_i64_20 = arith.constant 8 : i64
    %18 = arith.muli %17, %c8_i64_20 : i64
    %19 = arith.addi %16, %18 : i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    %base_buffer_21, %offset_22, %sizes_23:2, %strides_24:2 = memref.extract_strided_metadata %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %21 = arith.index_cast %strides_24#0 : index to i32
    %intptr_25 = memref.extract_aligned_pointer_as_index %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %22 = arith.index_cast %intptr_25 : index to i64
    %base_buffer_26, %offset_27, %sizes_28:2, %strides_29:2 = memref.extract_strided_metadata %subview_4 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %23 = arith.index_cast %offset_27 : index to i64
    %c8_i64_30 = arith.constant 8 : i64
    %24 = arith.muli %23, %c8_i64_30 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    call @polygeist_cublas_dsyr2k_lower(%7, %8, %arg2, %15, %9, %26, %21, %arg3, %20, %10) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %27 = bufferization.to_tensor %arg4 restrict writable : memref<?x?xf64>
    %28 = bufferization.to_memref %27 : memref<?x?xf64>
    memref.copy %28, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_dsyr2k_lower(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
}

