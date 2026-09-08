module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_mvt(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim) : memref<?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg1 : memref<?xf64>, memref<?xf64>
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg2, %c0_0 : memref<?xf64>
    %memref_2, %asyncToken_3 = gpu.alloc async [%1] (%dim_1) : memref<?xf64>
    %2 = gpu.memcpy async [%asyncToken_3] %memref_2, %arg2 : memref<?xf64>, memref<?xf64>
    %c0_4 = arith.constant 0 : index
    %dim_5 = memref.dim %arg3, %c0_4 : memref<?xf64>
    %memref_6, %asyncToken_7 = gpu.alloc async [%2] (%dim_5) : memref<?xf64>
    %3 = gpu.memcpy async [%asyncToken_7] %memref_6, %arg3 : memref<?xf64>, memref<?xf64>
    %c0_8 = arith.constant 0 : index
    %dim_9 = memref.dim %arg4, %c0_8 : memref<?xf64>
    %memref_10, %asyncToken_11 = gpu.alloc async [%3] (%dim_9) : memref<?xf64>
    %4 = gpu.memcpy async [%asyncToken_11] %memref_10, %arg4 : memref<?xf64>, memref<?xf64>
    %c0_12 = arith.constant 0 : index
    %dim_13 = memref.dim %arg5, %c0_12 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_14 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %memref_15, %asyncToken_16 = gpu.alloc async [%4] (%dim_13, %dim_14) : memref<?x?xf64>
    %5 = gpu.memcpy async [%asyncToken_16] %memref_15, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    gpu.wait [%5]
    %c1_17 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %6 = arith.index_cast %arg0 : i32 to index
    %dim_18 = memref.dim %memref_15, %c1_17 : memref<?x?xf64>
    %7 = arith.index_cast %dim_18 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_15 : memref<?x?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %intptr_19 = memref.extract_aligned_pointer_as_index %memref_6 : memref<?xf64> -> index
    %10 = arith.index_cast %intptr_19 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %subview = memref.subview %memref[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_20 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %12 = arith.index_cast %intptr_20 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %9, %7, %11, %cst, %13) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %dim_21 = memref.dim %memref_15, %c1_17 : memref<?x?xf64>
    %14 = arith.index_cast %dim_21 : index to i32
    %intptr_22 = memref.extract_aligned_pointer_as_index %memref_15 : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr_22 : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %intptr_23 = memref.extract_aligned_pointer_as_index %memref_10 : memref<?xf64> -> index
    %17 = arith.index_cast %intptr_23 : index to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %subview_24 = memref.subview %memref_2[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_25 = memref.extract_aligned_pointer_as_index %subview_24 : memref<?xf64, strided<[1]>> -> index
    %19 = arith.index_cast %intptr_25 : index to i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv_T(%arg0, %arg0, %cst, %16, %14, %18, %cst, %20) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %21 = gpu.wait async
    %22 = gpu.memcpy async [%21] %arg1, %memref : memref<?xf64>, memref<?xf64>
    %23 = gpu.dealloc async [%22] %memref : memref<?xf64>
    %24 = gpu.memcpy async [%23] %arg2, %memref_2 : memref<?xf64>, memref<?xf64>
    %25 = gpu.dealloc async [%24] %memref_2 : memref<?xf64>
    %26 = gpu.memcpy async [%25] %arg3, %memref_6 : memref<?xf64>, memref<?xf64>
    %27 = gpu.dealloc async [%26] %memref_6 : memref<?xf64>
    %28 = gpu.memcpy async [%27] %arg4, %memref_10 : memref<?xf64>, memref<?xf64>
    %29 = gpu.dealloc async [%28] %memref_10 : memref<?xf64>
    %30 = gpu.memcpy async [%29] %arg5, %memref_15 : memref<?x?xf64>, memref<?x?xf64>
    %31 = gpu.dealloc async [%30] %memref_15 : memref<?x?xf64>
    gpu.wait [%31]
    return
  }
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

