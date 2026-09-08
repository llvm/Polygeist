#map = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg4, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg4, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg4 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg5, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg5, %c1_3 : memref<?x?xf64>
    %memref_5, %asyncToken_6 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    %2 = gpu.memcpy async [%asyncToken_6] %memref_5, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg6, %c0_7 : memref<?x?xf64>
    %c1_9 = arith.constant 1 : index
    %dim_10 = memref.dim %arg6, %c1_9 : memref<?x?xf64>
    %memref_11, %asyncToken_12 = gpu.alloc async [%2] (%dim_8, %dim_10) : memref<?x?xf64>
    %3 = gpu.memcpy async [%asyncToken_12] %memref_11, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    gpu.wait [%3]
    %c8_i64 = arith.constant 8 : i64
    %c1_13 = arith.constant 1 : index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = arith.subi %5, %c1_13 : index
    %7 = affine.apply #map()[%6]
    %subview = memref.subview %memref_5[0, 0] [%7, %4] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_14 = memref.subview %memref_11[0, 0] [%5, %4] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %8 = arith.index_cast %7 : index to i32
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %base_buffer_15, %offset_16, %sizes_17:2, %strides_18:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %9 = arith.index_cast %strides#0 : index to i32
    %10 = arith.index_cast %strides_18#0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer_19, %offset_20, %sizes_21:2, %strides_22:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %offset_20 : index to i64
    %13 = arith.muli %12, %c8_i64 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_23 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %16 = arith.index_cast %intptr_23 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %base_buffer_24, %offset_25, %sizes_26:2, %strides_27:2 = memref.extract_strided_metadata %subview_14 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %18 = arith.index_cast %strides_27#0 : index to i32
    %intptr_28 = memref.extract_aligned_pointer_as_index %subview_14 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %19 = arith.index_cast %intptr_28 : index to i64
    %base_buffer_29, %offset_30, %sizes_31:2, %strides_32:2 = memref.extract_strided_metadata %subview_14 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %20 = arith.index_cast %offset_30 : index to i64
    %21 = arith.muli %20, %c8_i64 : i64
    %22 = arith.addi %19, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dsyr2k_lower(%8, %arg1, %arg2, %15, %9, %23, %18, %arg3, %17, %10) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %24 = gpu.wait async
    %25 = gpu.memcpy async [%24] %arg4, %memref : memref<?x?xf64>, memref<?x?xf64>
    %26 = gpu.dealloc async [%25] %memref : memref<?x?xf64>
    %27 = gpu.memcpy async [%26] %arg5, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %28 = gpu.dealloc async [%27] %memref_5 : memref<?x?xf64>
    %29 = gpu.memcpy async [%28] %arg6, %memref_11 : memref<?x?xf64>, memref<?x?xf64>
    %30 = gpu.dealloc async [%29] %memref_11 : memref<?x?xf64>
    gpu.wait [%30]
    return
  }
  func.func private @polygeist_cublas_dsyr2k_lower(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

