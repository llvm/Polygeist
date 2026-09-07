module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = 486721723013172567 : i64, polygeist.pipeline_scope_coalesced} {
    %c486721723013172567_i64 = arith.constant 486721723013172567 : i64
    call @polygeist_gpu_region_timing_begin(%c486721723013172567_i64) : (i64) -> ()
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
    %c0_21 = arith.constant 0 : index
    %dim_22 = memref.dim %arg7, %c0_21 : memref<?xf64>
    %c2_i32_23 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_23) : (i32) -> ()
    %memref_24, %asyncToken_25 = gpu.alloc async [%4] (%dim_22) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_26 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_26) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_25] %memref_24, %arg7 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%5]
    %c1_27 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0_28 = arith.constant 0 : index
    %6 = arith.index_cast %arg0 : i32 to index
    %dim_29 = memref.dim %memref_12, %c0_28 : memref<?xf64>
    %7 = arith.index_cast %dim_29 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_12 : memref<?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%7, %9) : (i32, !llvm.ptr) -> ()
    %dim_30 = memref.dim %memref_24, %c0_28 : memref<?xf64>
    %10 = arith.index_cast %dim_30 : index to i32
    %intptr_31 = memref.extract_aligned_pointer_as_index %memref_24 : memref<?xf64> -> index
    %11 = arith.index_cast %intptr_31 : index to i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_1d(%10, %12) : (i32, !llvm.ptr) -> ()
    %dim_32 = memref.dim %memref, %c1_27 : memref<?x?xf64>
    %13 = arith.index_cast %dim_32 : index to i32
    %intptr_33 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %14 = arith.index_cast %intptr_33 : index to i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_34 = memref.extract_aligned_pointer_as_index %memref_18 : memref<?xf64> -> index
    %16 = arith.index_cast %intptr_34 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %subview = memref.subview %memref_12[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_35 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %18 = arith.index_cast %intptr_35 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %15, %13, %17, %cst, %19) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_36 = memref.dim %memref_6, %c1_27 : memref<?x?xf64>
    %20 = arith.index_cast %dim_36 : index to i32
    %intptr_37 = memref.extract_aligned_pointer_as_index %memref_6 : memref<?x?xf64> -> index
    %21 = arith.index_cast %intptr_37 : index to i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %intptr_38 = memref.extract_aligned_pointer_as_index %memref_18 : memref<?xf64> -> index
    %23 = arith.index_cast %intptr_38 : index to i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %subview_39 = memref.subview %memref_24[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_40 = memref.extract_aligned_pointer_as_index %subview_39 : memref<?xf64, strided<[1]>> -> index
    %25 = arith.index_cast %intptr_40 : index to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %22, %20, %24, %cst, %26) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_41 = memref.dim %memref_24, %c0_28 : memref<?xf64>
    %27 = arith.index_cast %dim_41 : index to i32
    %intptr_42 = memref.extract_aligned_pointer_as_index %memref_12 : memref<?xf64> -> index
    %28 = arith.index_cast %intptr_42 : index to i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %intptr_43 = memref.extract_aligned_pointer_as_index %memref_24 : memref<?xf64> -> index
    %30 = arith.index_cast %intptr_43 : index to i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    call @polygeist_cublas_daxpby(%27, %arg1, %29, %arg2, %31) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %32 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %33 = gpu.memcpy async [%32] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %34 = gpu.dealloc async [%33] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_44 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_44) : (i32) -> ()
    %35 = gpu.memcpy async [%34] %arg4, %memref_6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_45 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_45) : (i32) -> ()
    %36 = gpu.dealloc async [%35] %memref_6 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_46 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_46) : (i32) -> ()
    %37 = gpu.memcpy async [%36] %arg5, %memref_12 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_47 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_47) : (i32) -> ()
    %38 = gpu.dealloc async [%37] %memref_12 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_48 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_48) : (i32) -> ()
    %39 = gpu.memcpy async [%38] %arg6, %memref_18 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_49 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_49) : (i32) -> ()
    %40 = gpu.dealloc async [%39] %memref_18 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_50 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_50) : (i32) -> ()
    %41 = gpu.memcpy async [%40] %arg7, %memref_24 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_51 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_51) : (i32) -> ()
    %42 = gpu.dealloc async [%41] %memref_24 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%42]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

