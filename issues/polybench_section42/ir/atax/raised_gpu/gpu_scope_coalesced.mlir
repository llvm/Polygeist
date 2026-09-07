module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = 6078541473191387945 : i64, polygeist.pipeline_scope_coalesced} {
    %c6078541473191387945_i64 = arith.constant 6078541473191387945 : i64
    call @polygeist_gpu_region_timing_begin(%c6078541473191387945_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg2, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg2 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg3, %c0_1 : memref<?xf64>
    %c2_i32_3 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_3) : (i32) -> ()
    %memref_4, %asyncToken_5 = gpu.alloc async [%1] (%dim_2) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_6 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_6) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_5] %memref_4, %arg3 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg4, %c0_7 : memref<?xf64>
    %c2_i32_9 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_9) : (i32) -> ()
    %memref_10, %asyncToken_11 = gpu.alloc async [%2] (%dim_8) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_12 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_12) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_11] %memref_10, %arg4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %arg5, %c0_13 : memref<?xf64>
    %c2_i32_15 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_15) : (i32) -> ()
    %memref_16, %asyncToken_17 = gpu.alloc async [%3] (%dim_14) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_18 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_18) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_17] %memref_16, %arg5 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%4]
    %c1_19 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0_20 = arith.constant 0 : index
    %5 = arith.index_cast %arg1 : i32 to index
    %dim_21 = memref.dim %memref_10, %c0_20 : memref<?xf64>
    %6 = arith.index_cast %dim_21 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_10 : memref<?xf64> -> index
    %7 = arith.index_cast %intptr : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%6, %8) : (i32, !llvm.ptr) -> ()
    %9 = arith.index_cast %arg0 : i32 to index
    %dim_22 = memref.dim %memref_16, %c0_20 : memref<?xf64>
    %10 = arith.index_cast %dim_22 : index to i32
    %intptr_23 = memref.extract_aligned_pointer_as_index %memref_16 : memref<?xf64> -> index
    %11 = arith.index_cast %intptr_23 : index to i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_1d(%10, %12) : (i32, !llvm.ptr) -> ()
    %dim_24 = memref.dim %memref, %c1_19 : memref<?x?xf64>
    %13 = arith.index_cast %dim_24 : index to i32
    %intptr_25 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %14 = arith.index_cast %intptr_25 : index to i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_26 = memref.extract_aligned_pointer_as_index %memref_4 : memref<?xf64> -> index
    %16 = arith.index_cast %intptr_26 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %subview = memref.subview %memref_16[0] [%9] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_27 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %18 = arith.index_cast %intptr_27 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg1, %cst, %15, %13, %17, %cst, %19) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_28 = memref.dim %memref, %c1_19 : memref<?x?xf64>
    %20 = arith.index_cast %dim_28 : index to i32
    %intptr_29 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %21 = arith.index_cast %intptr_29 : index to i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %intptr_30 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %23 = arith.index_cast %intptr_30 : index to i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %subview_31 = memref.subview %memref_10[0] [%5] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_32 = memref.extract_aligned_pointer_as_index %subview_31 : memref<?xf64, strided<[1]>> -> index
    %25 = arith.index_cast %intptr_32 : index to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv_T(%arg0, %arg1, %cst, %22, %20, %24, %cst, %26) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %27 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %28 = gpu.memcpy async [%27] %arg2, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %29 = gpu.dealloc async [%28] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_33 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_33) : (i32) -> ()
    %30 = gpu.memcpy async [%29] %arg3, %memref_4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_34 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_34) : (i32) -> ()
    %31 = gpu.dealloc async [%30] %memref_4 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_35 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_35) : (i32) -> ()
    %32 = gpu.memcpy async [%31] %arg4, %memref_10 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_36 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_36) : (i32) -> ()
    %33 = gpu.dealloc async [%32] %memref_10 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_37 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_37) : (i32) -> ()
    %34 = gpu.memcpy async [%33] %arg5, %memref_16 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_38 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_38) : (i32) -> ()
    %35 = gpu.dealloc async [%34] %memref_16 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%35]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

