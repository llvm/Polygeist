module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_mvt(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -7536468965854585798 : i64, polygeist.pipeline_scope_coalesced} {
    %c-7536468965854585798_i64 = arith.constant -7536468965854585798 : i64
    call @polygeist_gpu_region_timing_begin(%c-7536468965854585798_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg1 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg2, %c0_0 : memref<?xf64>
    %c2_i32_2 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_2) : (i32) -> ()
    %memref_3, %asyncToken_4 = gpu.alloc async [%1] (%dim_1) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_5 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_5) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_4] %memref_3, %arg2 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_6 = arith.constant 0 : index
    %dim_7 = memref.dim %arg3, %c0_6 : memref<?xf64>
    %c2_i32_8 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_8) : (i32) -> ()
    %memref_9, %asyncToken_10 = gpu.alloc async [%2] (%dim_7) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_11 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_11) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_10] %memref_9, %arg3 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_12 = arith.constant 0 : index
    %dim_13 = memref.dim %arg4, %c0_12 : memref<?xf64>
    %c2_i32_14 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_14) : (i32) -> ()
    %memref_15, %asyncToken_16 = gpu.alloc async [%3] (%dim_13) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_17 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_17) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_16] %memref_15, %arg4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_18 = arith.constant 0 : index
    %dim_19 = memref.dim %arg5, %c0_18 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_20 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %c2_i32_21 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_21) : (i32) -> ()
    %memref_22, %asyncToken_23 = gpu.alloc async [%4] (%dim_19, %dim_20) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_24 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_24) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_23] %memref_22, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%5]
    %c1_25 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %6 = arith.index_cast %arg0 : i32 to index
    %dim_26 = memref.dim %memref_22, %c1_25 : memref<?x?xf64>
    %7 = arith.index_cast %dim_26 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_22 : memref<?x?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %intptr_27 = memref.extract_aligned_pointer_as_index %memref_9 : memref<?xf64> -> index
    %10 = arith.index_cast %intptr_27 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %subview = memref.subview %memref[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_28 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %12 = arith.index_cast %intptr_28 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %9, %7, %11, %cst, %13) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_29 = memref.dim %memref_22, %c1_25 : memref<?x?xf64>
    %14 = arith.index_cast %dim_29 : index to i32
    %intptr_30 = memref.extract_aligned_pointer_as_index %memref_22 : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr_30 : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %intptr_31 = memref.extract_aligned_pointer_as_index %memref_15 : memref<?xf64> -> index
    %17 = arith.index_cast %intptr_31 : index to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %subview_32 = memref.subview %memref_3[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_33 = memref.extract_aligned_pointer_as_index %subview_32 : memref<?xf64, strided<[1]>> -> index
    %19 = arith.index_cast %intptr_33 : index to i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv_T(%arg0, %arg0, %cst, %16, %14, %18, %cst, %20) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %21 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %22 = gpu.memcpy async [%21] %arg1, %memref : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %23 = gpu.dealloc async [%22] %memref : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_34 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_34) : (i32) -> ()
    %24 = gpu.memcpy async [%23] %arg2, %memref_3 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_35 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_35) : (i32) -> ()
    %25 = gpu.dealloc async [%24] %memref_3 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_36 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_36) : (i32) -> ()
    %26 = gpu.memcpy async [%25] %arg3, %memref_9 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_37 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_37) : (i32) -> ()
    %27 = gpu.dealloc async [%26] %memref_9 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_38 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_38) : (i32) -> ()
    %28 = gpu.memcpy async [%27] %arg4, %memref_15 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_39 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_39) : (i32) -> ()
    %29 = gpu.dealloc async [%28] %memref_15 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_40 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_40) : (i32) -> ()
    %30 = gpu.memcpy async [%29] %arg5, %memref_22 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_41 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_41) : (i32) -> ()
    %31 = gpu.dealloc async [%30] %memref_22 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%31]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

