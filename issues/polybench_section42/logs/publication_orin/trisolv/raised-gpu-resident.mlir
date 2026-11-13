module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg1 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg2, %c0_1 : memref<?xf64>
    %memref_3, %asyncToken_4 = gpu.alloc async [%1] (%dim_2) : memref<?xf64>
    %2 = gpu.memcpy async [%asyncToken_4] %memref_3, %arg2 : memref<?xf64>, memref<?xf64>
    %c0_5 = arith.constant 0 : index
    %dim_6 = memref.dim %arg3, %c0_5 : memref<?xf64>
    %memref_7, %asyncToken_8 = gpu.alloc async [%2] (%dim_6) : memref<?xf64>
    %3 = gpu.memcpy async [%asyncToken_8] %memref_7, %arg3 : memref<?xf64>, memref<?xf64>
    gpu.wait [%3]
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %memref_3, %c0_9 : memref<?xf64>
    %4 = arith.index_cast %dim_10 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %5 = arith.index_cast %intptr : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %intptr_11 = memref.extract_aligned_pointer_as_index %memref_7 : memref<?xf64> -> index
    %7 = arith.index_cast %intptr_11 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %intptr_12 = memref.extract_aligned_pointer_as_index %memref_3 : memref<?xf64> -> index
    %9 = arith.index_cast %intptr_12 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dtrsv_lower_row_major(%4, %6, %8, %10) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %11 = gpu.wait async
    %12 = gpu.memcpy async [%11] %arg1, %memref : memref<?x?xf64>, memref<?x?xf64>
    %13 = gpu.dealloc async [%12] %memref : memref<?x?xf64>
    %14 = gpu.memcpy async [%13] %arg2, %memref_3 : memref<?xf64>, memref<?xf64>
    %15 = gpu.dealloc async [%14] %memref_3 : memref<?xf64>
    %16 = gpu.memcpy async [%15] %arg3, %memref_7 : memref<?xf64>, memref<?xf64>
    %17 = gpu.dealloc async [%16] %memref_7 : memref<?xf64>
    gpu.wait [%17]
    return
  }
  func.func private @polygeist_cublas_dtrsv_lower_row_major(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

