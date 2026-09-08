module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -3006860672589866865 : i64, polygeist.pipeline_scope_coalesced} {
    %c-3006860672589866865_i64 = arith.constant -3006860672589866865 : i64
    call @polygeist_gpu_region_timing_begin(%c-3006860672589866865_i64) : (i64) -> ()
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
    %c0_15 = arith.constant 0 : index
    %dim_16 = memref.dim %arg6, %c0_15 : memref<?xf64>
    %c2_i32_17 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_17) : (i32) -> ()
    %memref_18, %asyncToken_19 = gpu.alloc async [%3] (%dim_16) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_20 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_20) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_19] %memref_18, %arg6 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%4]
    %c1_21 = arith.constant 1 : index
    %c0_22 = arith.constant 0 : index
    %dim_23 = memref.dim %memref, %c0_22 : memref<?x?xf64>
    %5 = arith.index_cast %dim_23 : index to i32
    %dim_24 = memref.dim %memref, %c1_21 : memref<?x?xf64>
    %6 = arith.index_cast %dim_24 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %7 = arith.index_cast %intptr : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %9 = arith.index_cast %strides#0 : index to i32
    %intptr_25 = memref.extract_aligned_pointer_as_index %memref_6 : memref<?x?xf64> -> index
    %10 = arith.index_cast %intptr_25 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %base_buffer_26, %offset_27, %sizes_28:2, %strides_29:2 = memref.extract_strided_metadata %memref_6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %strides_29#0 : index to i32
    %intptr_30 = memref.extract_aligned_pointer_as_index %memref_12 : memref<?xf64> -> index
    %13 = arith.index_cast %intptr_30 : index to i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %intptr_31 = memref.extract_aligned_pointer_as_index %memref_18 : memref<?xf64> -> index
    %15 = arith.index_cast %intptr_31 : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dcorrelation_row_major(%5, %6, %arg2, %8, %9, %11, %12, %14, %16) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %17 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %18 = gpu.memcpy async [%17] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %19 = gpu.dealloc async [%18] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_32) : (i32) -> ()
    %20 = gpu.memcpy async [%19] %arg4, %memref_6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_33 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_33) : (i32) -> ()
    %21 = gpu.dealloc async [%20] %memref_6 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_34 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_34) : (i32) -> ()
    %22 = gpu.memcpy async [%21] %arg5, %memref_12 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_35 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_35) : (i32) -> ()
    %23 = gpu.dealloc async [%22] %memref_12 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_36 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_36) : (i32) -> ()
    %24 = gpu.memcpy async [%23] %arg6, %memref_18 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_37 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_37) : (i32) -> ()
    %25 = gpu.dealloc async [%24] %memref_18 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%25]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dcorrelation_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

