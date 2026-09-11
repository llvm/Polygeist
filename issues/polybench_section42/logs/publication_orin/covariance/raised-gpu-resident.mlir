module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg3 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg4, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg4, %c1_3 : memref<?x?xf64>
    %memref_5, %asyncToken_6 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    %2 = gpu.memcpy async [%asyncToken_6] %memref_5, %arg4 : memref<?x?xf64>, memref<?x?xf64>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg5, %c0_7 : memref<?xf64>
    %memref_9, %asyncToken_10 = gpu.alloc async [%2] (%dim_8) : memref<?xf64>
    %3 = gpu.memcpy async [%asyncToken_10] %memref_9, %arg5 : memref<?xf64>, memref<?xf64>
    gpu.wait [%3]
    %c1_11 = arith.constant 1 : index
    %c0_12 = arith.constant 0 : index
    %dim_13 = memref.dim %memref, %c0_12 : memref<?x?xf64>
    %4 = arith.index_cast %dim_13 : index to i32
    %dim_14 = memref.dim %memref, %c1_11 : memref<?x?xf64>
    %5 = arith.index_cast %dim_14 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %8 = arith.index_cast %strides#0 : index to i32
    %intptr_15 = memref.extract_aligned_pointer_as_index %memref_5 : memref<?x?xf64> -> index
    %9 = arith.index_cast %intptr_15 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %base_buffer_16, %offset_17, %sizes_18:2, %strides_19:2 = memref.extract_strided_metadata %memref_5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %11 = arith.index_cast %strides_19#0 : index to i32
    %intptr_20 = memref.extract_aligned_pointer_as_index %memref_9 : memref<?xf64> -> index
    %12 = arith.index_cast %intptr_20 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dcovariance_row_major(%4, %5, %arg2, %7, %8, %10, %11, %13) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %14 = gpu.wait async
    %15 = gpu.memcpy async [%14] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    %16 = gpu.dealloc async [%15] %memref : memref<?x?xf64>
    %17 = gpu.memcpy async [%16] %arg4, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %18 = gpu.dealloc async [%17] %memref_5 : memref<?x?xf64>
    %19 = gpu.memcpy async [%18] %arg5, %memref_9 : memref<?xf64>, memref<?xf64>
    %20 = gpu.dealloc async [%19] %memref_9 : memref<?xf64>
    gpu.wait [%20]
    return
  }
  func.func private @polygeist_cublas_dcovariance_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

