module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -8212070572546283992 : i64, polygeist.pipeline_scope_coalesced} {
    %c-8212070572546283992_i64 = arith.constant -8212070572546283992 : i64
    call @polygeist_gpu_region_timing_begin(%c-8212070572546283992_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg3 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg4, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg4, %c1_3 : memref<?x?xf64>
    %c2_i32_5 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_5) : (i32) -> ()
    %memref_6, %asyncToken_7 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_8 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_8) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_7] %memref_6, %arg4 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg5, %c0_9 : memref<?xf64>
    %c2_i32_11 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_11) : (i32) -> ()
    %memref_12, %asyncToken_13 = gpu.alloc async [%2] (%dim_10) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_14 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_14) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_13] %memref_12, %arg5 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%3]
    %c1_15 = arith.constant 1 : index
    %c0_16 = arith.constant 0 : index
    %dim_17 = memref.dim %memref, %c0_16 : memref<?x?xf64>
    %4 = arith.index_cast %dim_17 : index to i32
    %dim_18 = memref.dim %memref, %c1_15 : memref<?x?xf64>
    %5 = arith.index_cast %dim_18 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %8 = arith.index_cast %strides#0 : index to i32
    %intptr_19 = memref.extract_aligned_pointer_as_index %memref_6 : memref<?x?xf64> -> index
    %9 = arith.index_cast %intptr_19 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %base_buffer_20, %offset_21, %sizes_22:2, %strides_23:2 = memref.extract_strided_metadata %memref_6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %11 = arith.index_cast %strides_23#0 : index to i32
    %intptr_24 = memref.extract_aligned_pointer_as_index %memref_12 : memref<?xf64> -> index
    %12 = arith.index_cast %intptr_24 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dcovariance_row_major(%4, %5, %arg2, %7, %8, %10, %11, %13) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %14 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %15 = gpu.memcpy async [%14] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %16 = gpu.dealloc async [%15] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_25 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_25) : (i32) -> ()
    %17 = gpu.memcpy async [%16] %arg4, %memref_6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_26 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_26) : (i32) -> ()
    %18 = gpu.dealloc async [%17] %memref_6 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_27 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_27) : (i32) -> ()
    %19 = gpu.memcpy async [%18] %arg5, %memref_12 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_28 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_28) : (i32) -> ()
    %20 = gpu.dealloc async [%19] %memref_12 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%20]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dcovariance_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

