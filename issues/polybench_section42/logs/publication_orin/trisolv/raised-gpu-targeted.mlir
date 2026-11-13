module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = 1331154239229117518 : i64, polygeist.pipeline_scope_coalesced} {
    %c1331154239229117518_i64 = arith.constant 1331154239229117518 : i64
    call @polygeist_gpu_region_timing_begin(%c1331154239229117518_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg1 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg2, %c0_1 : memref<?xf64>
    %c2_i32_3 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_3) : (i32) -> ()
    %memref_4, %asyncToken_5 = gpu.alloc async [%1] (%dim_2) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_6 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_6) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_5] %memref_4, %arg2 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg3, %c0_7 : memref<?xf64>
    %c2_i32_9 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_9) : (i32) -> ()
    %memref_10, %asyncToken_11 = gpu.alloc async [%2] (%dim_8) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_12 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_12) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_11] %memref_10, %arg3 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%3]
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %memref_4, %c0_13 : memref<?xf64>
    %4 = arith.index_cast %dim_14 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %5 = arith.index_cast %intptr : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %intptr_15 = memref.extract_aligned_pointer_as_index %memref_10 : memref<?xf64> -> index
    %7 = arith.index_cast %intptr_15 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %intptr_16 = memref.extract_aligned_pointer_as_index %memref_4 : memref<?xf64> -> index
    %9 = arith.index_cast %intptr_16 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dtrsv_lower_row_major(%4, %6, %8, %10) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %11 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %12 = gpu.memcpy async [%11] %arg1, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %13 = gpu.dealloc async [%12] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_17 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_17) : (i32) -> ()
    %14 = gpu.memcpy async [%13] %arg2, %memref_4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_18 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_18) : (i32) -> ()
    %15 = gpu.dealloc async [%14] %memref_4 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_19 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_19) : (i32) -> ()
    %16 = gpu.memcpy async [%15] %arg3, %memref_10 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_20 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_20) : (i32) -> ()
    %17 = gpu.dealloc async [%16] %memref_10 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%17]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dtrsv_lower_row_major(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

