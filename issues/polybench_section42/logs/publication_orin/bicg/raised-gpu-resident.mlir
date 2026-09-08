module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg2, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg2 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg3, %c0_1 : memref<?xf64>
    %memref_3, %asyncToken_4 = gpu.alloc async [%1] (%dim_2) : memref<?xf64>
    %2 = gpu.memcpy async [%asyncToken_4] %memref_3, %arg3 : memref<?xf64>, memref<?xf64>
    %c0_5 = arith.constant 0 : index
    %dim_6 = memref.dim %arg4, %c0_5 : memref<?xf64>
    %memref_7, %asyncToken_8 = gpu.alloc async [%2] (%dim_6) : memref<?xf64>
    %3 = gpu.memcpy async [%asyncToken_8] %memref_7, %arg4 : memref<?xf64>, memref<?xf64>
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg5, %c0_9 : memref<?xf64>
    %memref_11, %asyncToken_12 = gpu.alloc async [%3] (%dim_10) : memref<?xf64>
    %4 = gpu.memcpy async [%asyncToken_12] %memref_11, %arg5 : memref<?xf64>, memref<?xf64>
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %arg6, %c0_13 : memref<?xf64>
    %memref_15, %asyncToken_16 = gpu.alloc async [%4] (%dim_14) : memref<?xf64>
    %5 = gpu.memcpy async [%asyncToken_16] %memref_15, %arg6 : memref<?xf64>, memref<?xf64>
    gpu.wait [%5]
    %c1_17 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0_18 = arith.constant 0 : index
    %6 = arith.index_cast %arg0 : i32 to index
    %dim_19 = memref.dim %memref_3, %c0_18 : memref<?xf64>
    %7 = arith.index_cast %dim_19 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_3 : memref<?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%7, %9) : (i32, !llvm.ptr) -> ()
    %10 = arith.index_cast %arg1 : i32 to index
    %dim_20 = memref.dim %memref_7, %c0_18 : memref<?xf64>
    %11 = arith.index_cast %dim_20 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_21 = memref.extract_aligned_pointer_as_index %memref_7 : memref<?xf64> -> index
    %12 = arith.index_cast %intptr_21 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%11, %13) : (i32, !llvm.ptr) -> ()
    %dim_22 = memref.dim %memref, %c1_17 : memref<?x?xf64>
    %14 = arith.index_cast %dim_22 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_23 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr_23 : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %intptr_24 = memref.extract_aligned_pointer_as_index %memref_15 : memref<?xf64> -> index
    %17 = arith.index_cast %intptr_24 : index to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %subview = memref.subview %memref_3[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_25 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %19 = arith.index_cast %intptr_25 : index to i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%arg1, %arg0, %cst, %16, %14, %18, %cst, %20) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_26 = memref.dim %memref, %c1_17 : memref<?x?xf64>
    %21 = arith.index_cast %dim_26 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_27 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %22 = arith.index_cast %intptr_27 : index to i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %intptr_28 = memref.extract_aligned_pointer_as_index %memref_11 : memref<?xf64> -> index
    %24 = arith.index_cast %intptr_28 : index to i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    %subview_29 = memref.subview %memref_7[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_30 = memref.extract_aligned_pointer_as_index %subview_29 : memref<?xf64, strided<[1]>> -> index
    %26 = arith.index_cast %intptr_30 : index to i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg1, %arg0, %cst, %23, %21, %25, %cst, %27) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %28 = gpu.wait async
    %29 = gpu.memcpy async [%28] %arg2, %memref : memref<?x?xf64>, memref<?x?xf64>
    %30 = gpu.dealloc async [%29] %memref : memref<?x?xf64>
    %31 = gpu.memcpy async [%30] %arg3, %memref_3 : memref<?xf64>, memref<?xf64>
    %32 = gpu.dealloc async [%31] %memref_3 : memref<?xf64>
    %33 = gpu.memcpy async [%32] %arg4, %memref_7 : memref<?xf64>, memref<?xf64>
    %34 = gpu.dealloc async [%33] %memref_7 : memref<?xf64>
    %35 = gpu.memcpy async [%34] %arg5, %memref_11 : memref<?xf64>, memref<?xf64>
    %36 = gpu.dealloc async [%35] %memref_11 : memref<?xf64>
    %37 = gpu.memcpy async [%36] %arg6, %memref_15 : memref<?xf64>, memref<?xf64>
    %38 = gpu.dealloc async [%37] %memref_15 : memref<?xf64>
    gpu.wait [%38]
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

