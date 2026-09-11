#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %5 = affine.apply #map(%4)
    %extracted_slice = tensor.extract_slice %1[0, 0] [%3, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%5, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0] [%3, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %subview = memref.subview %arg5[0, 0] [%3, %2] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %subview, %c0 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %6 = arith.index_cast %dim : index to i32
    %c1_2 = arith.constant 1 : index
    %dim_3 = memref.dim %subview, %c1_2 : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %7 = arith.index_cast %dim_3 : index to i32
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %8 = arith.index_cast %strides#0 : index to i32
    %9 = arith.index_cast %strides_7#0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %10 = arith.index_cast %intptr : index to i64
    %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %11 = arith.index_cast %offset_9 : index to i64
    %c8_i64 = arith.constant 8 : i64
    %12 = arith.muli %11, %c8_i64 : i64
    %13 = arith.addi %10, %12 : i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %intptr_12 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr_12 : index to i64
    %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %16 = arith.index_cast %offset_14 : index to i64
    %c8_i64_17 = arith.constant 8 : i64
    %17 = arith.muli %16, %c8_i64_17 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    call @polygeist_cublas_dsyrk_lower(%6, %7, %arg2, %14, %8, %arg3, %19, %9) : (i32, i32, f64, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %20 = bufferization.to_tensor %arg4 restrict writable : memref<?x?xf64>
    %21 = bufferization.to_memref %20 : memref<?x?xf64>
    memref.copy %21, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_dsyrk_lower(i32, i32, f64, !llvm.ptr, i32, f64, !llvm.ptr, i32)
}

