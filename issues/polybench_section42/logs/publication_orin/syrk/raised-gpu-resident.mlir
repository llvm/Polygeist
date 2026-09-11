module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg4, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg4, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg4 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg5, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg5, %c1_3 : memref<?x?xf64>
    %memref_5, %asyncToken_6 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    %2 = gpu.memcpy async [%asyncToken_6] %memref_5, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    gpu.wait [%2]
    %c8_i64 = arith.constant 8 : i64
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %subview = memref.subview %memref_5[0, 0] [%4, %3] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %base_buffer_7, %offset_8, %sizes_9:2, %strides_10:2 = memref.extract_strided_metadata %memref : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %5 = arith.index_cast %strides#0 : index to i32
    %6 = arith.index_cast %strides_10#0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %7 = arith.index_cast %intptr : index to i64
    %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %8 = arith.index_cast %offset_12 : index to i64
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = arith.addi %7, %9 : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %intptr_15 = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %12 = arith.index_cast %intptr_15 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dsyrk_lower(%arg0, %arg1, %arg2, %11, %5, %arg3, %13, %6) : (i32, i32, f64, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %14 = gpu.wait async
    %15 = gpu.memcpy async [%14] %arg4, %memref : memref<?x?xf64>, memref<?x?xf64>
    %16 = gpu.dealloc async [%15] %memref : memref<?x?xf64>
    %17 = gpu.memcpy async [%16] %arg5, %memref_5 : memref<?x?xf64>, memref<?x?xf64>
    %18 = gpu.dealloc async [%17] %memref_5 : memref<?x?xf64>
    gpu.wait [%18]
    return
  }
  func.func private @polygeist_cublas_dsyrk_lower(i32, i32, f64, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

