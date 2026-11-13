module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
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
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    %dim_15 = memref.dim %memref, %c0_13 : memref<?x?xf64>
    %4 = arith.index_cast %dim_15 : index to i32
    %dim_16 = memref.dim %memref, %c1_14 : memref<?x?xf64>
    %5 = arith.index_cast %dim_16 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_5 : memref<?x?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %memref_5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %8 = arith.index_cast %strides#0 : index to i32
    %intptr_17 = memref.extract_aligned_pointer_as_index %memref_11 : memref<?x?xf64> -> index
    %9 = arith.index_cast %intptr_17 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %base_buffer_18, %offset_19, %sizes_20:2, %strides_21:2 = memref.extract_strided_metadata %memref_11 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %11 = arith.index_cast %strides_21#0 : index to i32
    %intptr_22 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %12 = arith.index_cast %intptr_22 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %base_buffer_23, %offset_24, %sizes_25:2, %strides_26:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %14 = arith.index_cast %strides_26#0 : index to i32
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dsymm_left_lower_row_major(%4, %5, %arg2, %7, %8, %10, %11, %arg3, %13, %14) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %15 = gpu.wait async
    %16 = gpu.memcpy async [%15] %arg4, %memref : memref<?x?xf64>, memref<?x?xf64>
    %17 = gpu.dealloc async [%16] %memref : memref<?x?xf64>
    %18 = gpu.memcpy async [%17] %arg5, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %19 = gpu.dealloc async [%18] %memref_5 : memref<?x?xf64>
    %20 = gpu.memcpy async [%19] %arg6, %memref_11 : memref<?x?xf64>, memref<?x?xf64>
    %21 = gpu.dealloc async [%20] %memref_11 : memref<?x?xf64>
    gpu.wait [%21]
    return
  }
  func.func private @polygeist_cublas_dsymm_left_lower_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

