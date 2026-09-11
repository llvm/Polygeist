module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg3 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg4, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg4, %c1_3 : memref<?x?xf64>
    %memref_5, %asyncToken_6 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    %2 = gpu.memcpy async [%asyncToken_6] %memref_5, %arg4 : memref<?x?xf64>, memref<?x?xf64>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg5, %c0_7 : memref<?xf64>
    %memref_9, %asyncToken_10 = gpu.alloc async [%2] (%dim_8) : memref<?xf64>
    %3 = gpu.memcpy async [%asyncToken_10] %memref_9, %arg5 : memref<?xf64>, memref<?xf64>
    %c0_11 = arith.constant 0 : index
    %dim_12 = memref.dim %arg6, %c0_11 : memref<?xf64>
    %memref_13, %asyncToken_14 = gpu.alloc async [%3] (%dim_12) : memref<?xf64>
    %4 = gpu.memcpy async [%asyncToken_14] %memref_13, %arg6 : memref<?xf64>, memref<?xf64>
    %c0_15 = arith.constant 0 : index
    %dim_16 = memref.dim %arg7, %c0_15 : memref<?xf64>
    %memref_17, %asyncToken_18 = gpu.alloc async [%4] (%dim_16) : memref<?xf64>
    %5 = gpu.memcpy async [%asyncToken_18] %memref_17, %arg7 : memref<?xf64>, memref<?xf64>
    gpu.wait [%5]
    %c1_19 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0_20 = arith.constant 0 : index
    %6 = arith.index_cast %arg0 : i32 to index
    %dim_21 = memref.dim %memref_9, %c0_20 : memref<?xf64>
    %7 = arith.index_cast %dim_21 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref_9 : memref<?xf64> -> index
    %8 = arith.index_cast %intptr : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%7, %9) : (i32, !llvm.ptr) -> ()
    %dim_22 = memref.dim %memref_17, %c0_20 : memref<?xf64>
    %10 = arith.index_cast %dim_22 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_23 = memref.extract_aligned_pointer_as_index %memref_17 : memref<?xf64> -> index
    %11 = arith.index_cast %intptr_23 : index to i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%10, %12) : (i32, !llvm.ptr) -> ()
    %dim_24 = memref.dim %memref, %c1_19 : memref<?x?xf64>
    %13 = arith.index_cast %dim_24 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_25 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %14 = arith.index_cast %intptr_25 : index to i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_26 = memref.extract_aligned_pointer_as_index %memref_13 : memref<?xf64> -> index
    %16 = arith.index_cast %intptr_26 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %subview = memref.subview %memref_9[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_27 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %18 = arith.index_cast %intptr_27 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %15, %13, %17, %cst, %19) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %dim_28 = memref.dim %memref_5, %c1_19 : memref<?x?xf64>
    %20 = arith.index_cast %dim_28 : index to i32
    %intptr_29 = memref.extract_aligned_pointer_as_index %memref_5 : memref<?x?xf64> -> index
    %21 = arith.index_cast %intptr_29 : index to i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %intptr_30 = memref.extract_aligned_pointer_as_index %memref_13 : memref<?xf64> -> index
    %23 = arith.index_cast %intptr_30 : index to i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %subview_31 = memref.subview %memref_17[0] [%6] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_32 = memref.extract_aligned_pointer_as_index %subview_31 : memref<?xf64, strided<[1]>> -> index
    %25 = arith.index_cast %intptr_32 : index to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %22, %20, %24, %cst, %26) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_33 = memref.dim %memref_17, %c0_20 : memref<?xf64>
    %27 = arith.index_cast %dim_33 : index to i32
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_34 = memref.extract_aligned_pointer_as_index %memref_9 : memref<?xf64> -> index
    %28 = arith.index_cast %intptr_34 : index to i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %intptr_35 = memref.extract_aligned_pointer_as_index %memref_17 : memref<?xf64> -> index
    %30 = arith.index_cast %intptr_35 : index to i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_daxpby(%27, %arg1, %29, %arg2, %31) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %32 = gpu.wait async
    %33 = gpu.memcpy async [%32] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    %34 = gpu.dealloc async [%33] %memref : memref<?x?xf64>
    %35 = gpu.memcpy async [%34] %arg4, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %36 = gpu.dealloc async [%35] %memref_5 : memref<?x?xf64>
    %37 = gpu.memcpy async [%36] %arg5, %memref_9 : memref<?xf64>, memref<?xf64>
    %38 = gpu.dealloc async [%37] %memref_9 : memref<?xf64>
    %39 = gpu.memcpy async [%38] %arg6, %memref_13 : memref<?xf64>, memref<?xf64>
    %40 = gpu.dealloc async [%39] %memref_13 : memref<?xf64>
    %41 = gpu.memcpy async [%40] %arg7, %memref_17 : memref<?xf64>, memref<?xf64>
    %42 = gpu.dealloc async [%41] %memref_17 : memref<?xf64>
    gpu.wait [%42]
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

