module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -2281302616747103834 : i64, polygeist.pipeline_scope_coalesced} {
    %c-2281302616747103834_i64 = arith.constant -2281302616747103834 : i64
    call @polygeist_gpu_region_timing_begin(%c-2281302616747103834_i64) : (i64) -> ()
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
    %c0_19 = arith.constant 0 : index
    %dim_20 = memref.dim %arg6, %c0_19 : memref<?xf64>
    %c2_i32_21 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_21) : (i32) -> ()
    %memref_22, %asyncToken_23 = gpu.alloc async [%4] (%dim_20) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_24 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_24) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_23] %memref_22, %arg6 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%5]
    %c1_25 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0_26 = arith.constant 0 : index
    %6 = arith.index_cast %arg0 : i32 to index
    %dim_27 = memref.dim %memref_4, %c0_26 : memref<?xf64>
    %7 = arith.index_cast %dim_27 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_4 : memref<?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%7, %9) : (i32, !llvm.ptr) -> ()
    %10 = arith.index_cast %arg1 : i32 to index
    %dim_28 = memref.dim %memref_10, %c0_26 : memref<?xf64>
    %11 = arith.index_cast %dim_28 : index to i32
    %intptr_29 = memref.extract_aligned_pointer_as_index %memref_10 : memref<?xf64> -> index
    %12 = arith.index_cast %intptr_29 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_1d(%11, %13) : (i32, !llvm.ptr) -> ()
    %dim_30 = memref.dim %memref, %c1_25 : memref<?x?xf64>
    %14 = arith.index_cast %dim_30 : index to i32
    %intptr_31 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr_31 : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %intptr_32 = memref.extract_aligned_pointer_as_index %memref_22 : memref<?xf64> -> index
    %17 = arith.index_cast %intptr_32 : index to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %subview = memref.subview %memref_4[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_33 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %19 = arith.index_cast %intptr_33 : index to i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv_T(%arg1, %arg0, %cst, %16, %14, %18, %cst, %20) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_34 = memref.dim %memref, %c1_25 : memref<?x?xf64>
    %21 = arith.index_cast %dim_34 : index to i32
    %intptr_35 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %22 = arith.index_cast %intptr_35 : index to i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %intptr_36 = memref.extract_aligned_pointer_as_index %memref_16 : memref<?xf64> -> index
    %24 = arith.index_cast %intptr_36 : index to i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    %subview_37 = memref.subview %memref_10[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_38 = memref.extract_aligned_pointer_as_index %subview_37 : memref<?xf64, strided<[1]>> -> index
    %26 = arith.index_cast %intptr_38 : index to i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg1, %arg0, %cst, %23, %21, %25, %cst, %27) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %28 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %29 = gpu.memcpy async [%28] %arg2, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %30 = gpu.dealloc async [%29] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_39 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_39) : (i32) -> ()
    %31 = gpu.memcpy async [%30] %arg3, %memref_4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_40 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_40) : (i32) -> ()
    %32 = gpu.dealloc async [%31] %memref_4 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_41 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_41) : (i32) -> ()
    %33 = gpu.memcpy async [%32] %arg4, %memref_10 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_42 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_42) : (i32) -> ()
    %34 = gpu.dealloc async [%33] %memref_10 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_43 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_43) : (i32) -> ()
    %35 = gpu.memcpy async [%34] %arg5, %memref_16 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_44 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_44) : (i32) -> ()
    %36 = gpu.dealloc async [%35] %memref_16 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_45 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_45) : (i32) -> ()
    %37 = gpu.memcpy async [%36] %arg6, %memref_22 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_46 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_46) : (i32) -> ()
    %38 = gpu.dealloc async [%37] %memref_22 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%38]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

