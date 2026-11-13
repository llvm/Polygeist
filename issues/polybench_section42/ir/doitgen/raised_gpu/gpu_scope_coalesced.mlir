module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -5973564920580309495 : i64, polygeist.pipeline_scope_coalesced} {
    %c-5973564920580309495_i64 = arith.constant -5973564920580309495 : i64
    call @polygeist_gpu_region_timing_begin(%c-5973564920580309495_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?x?xf64>
    %c2 = arith.constant 2 : index
    %dim_1 = memref.dim %arg3, %c2 : memref<?x?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0, %dim_1) : memref<?x?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg3 : memref<?x?x?xf64>, memref<?x?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_2 = arith.constant 0 : index
    %dim_3 = memref.dim %arg4, %c0_2 : memref<?x?xf64>
    %c1_4 = arith.constant 1 : index
    %dim_5 = memref.dim %arg4, %c1_4 : memref<?x?xf64>
    %c2_i32_6 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_6) : (i32) -> ()
    %memref_7, %asyncToken_8 = gpu.alloc async [%1] (%dim_3, %dim_5) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_9 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_9) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_8] %memref_7, %arg4 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_10 = arith.constant 0 : index
    %dim_11 = memref.dim %arg5, %c0_10 : memref<?xf64>
    %c2_i32_12 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_12) : (i32) -> ()
    %memref_13, %asyncToken_14 = gpu.alloc async [%2] (%dim_11) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_15 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_15) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_14] %memref_13, %arg5 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%3]
    %c8_i64 = arith.constant 8 : i64
    %c1_16 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %cst_17 = arith.constant 0.000000e+00 : f64
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    affine.for %arg6 = 0 to %6 {
      affine.for %arg7 = 0 to %4 {
        %dim_22 = memref.dim %memref_7, %c1_16 : memref<?x?xf64>
        %14 = arith.index_cast %dim_22 : index to i32
        %intptr = memref.extract_aligned_pointer_as_index %memref_7 : memref<?x?xf64> -> index
        %15 = arith.index_cast %intptr : index to i64
        %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
        %intptr_23 = memref.extract_aligned_pointer_as_index %memref : memref<?x?x?xf64> -> index
        %17 = arith.index_cast %intptr_23 : index to i64
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %memref : memref<?x?x?xf64> -> memref<f64>, index, index, index, index, index, index, index
        %18 = arith.index_cast %arg6 : index to i64
        %19 = arith.index_cast %strides#0 : index to i64
        %20 = arith.muli %18, %19 : i64
        %21 = arith.index_cast %arg7 : index to i64
        %22 = arith.index_cast %strides#1 : index to i64
        %23 = arith.muli %21, %22 : i64
        %24 = arith.addi %20, %23 : i64
        %25 = arith.muli %24, %c8_i64 : i64
        %26 = arith.addi %17, %25 : i64
        %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
        %subview = memref.subview %memref[%arg6, %arg7, 0] [1, 1, %5] [1, 1, 1] : memref<?x?x?xf64> to memref<?xf64, strided<[1], offset: ?>>
        %intptr_24 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1], offset: ?>> -> index
        %28 = arith.index_cast %intptr_24 : index to i64
        %base_buffer_25, %offset_26, %sizes_27, %strides_28 = memref.extract_strided_metadata %subview : memref<?xf64, strided<[1], offset: ?>> -> memref<f64>, index, index, index
        %29 = arith.index_cast %offset_26 : index to i64
        %30 = arith.muli %29, %c8_i64 : i64
        %31 = arith.addi %28, %30 : i64
        %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
        func.call @polygeist_cublas_dgemv_T(%arg2, %arg2, %cst, %16, %14, %27, %cst_17, %32) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
      }
    }
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %7 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %8 = gpu.memcpy async [%7] %arg3, %memref : memref<?x?x?xf64>, memref<?x?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %9 = gpu.dealloc async [%8] %memref : memref<?x?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_18 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_18) : (i32) -> ()
    %10 = gpu.memcpy async [%9] %arg4, %memref_7 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_19 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_19) : (i32) -> ()
    %11 = gpu.dealloc async [%10] %memref_7 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_20 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_20) : (i32) -> ()
    %12 = gpu.memcpy async [%11] %arg5, %memref_13 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_21 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_21) : (i32) -> ()
    %13 = gpu.dealloc async [%12] %memref_13 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%13]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
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

