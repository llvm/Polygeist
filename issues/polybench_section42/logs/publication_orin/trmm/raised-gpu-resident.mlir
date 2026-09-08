module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
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
    gpu.wait [%2]
    %c1_7 = arith.constant 1 : index
    %c0_8 = arith.constant 0 : index
    %dim_9 = memref.dim %memref_5, %c0_8 : memref<?x?xf64>
    %3 = arith.index_cast %dim_9 : index to i32
    %dim_10 = memref.dim %memref_5, %c1_7 : memref<?x?xf64>
    %4 = arith.index_cast %dim_10 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %5 = arith.index_cast %intptr : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %7 = arith.index_cast %strides#0 : index to i32
    %intptr_11 = memref.extract_aligned_pointer_as_index %memref_5 : memref<?x?xf64> -> index
    %8 = arith.index_cast %intptr_11 : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %memref_5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %10 = arith.index_cast %strides_15#0 : index to i32
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dtrmm_left_lower_trans_unit_row_major(%3, %4, %arg2, %6, %7, %9, %10) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %11 = gpu.wait async
    %12 = gpu.memcpy async [%11] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    %13 = gpu.dealloc async [%12] %memref : memref<?x?xf64>
    %14 = gpu.memcpy async [%13] %arg4, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %15 = gpu.dealloc async [%14] %memref_5 : memref<?x?xf64>
    gpu.wait [%15]
    return
  }
  func.func private @polygeist_cublas_dtrmm_left_lower_trans_unit_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

