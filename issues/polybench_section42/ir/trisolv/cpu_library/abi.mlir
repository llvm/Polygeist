module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 restrict : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 restrict : memref<?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0 : memref<?xf64>
    %5 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<?x?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %7 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %8 = arith.muli %7, %c8_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %intptr_0 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %11 = arith.index_cast %intptr_0 : index to i64
    %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %arg3 : memref<?xf64> -> memref<f64>, index, index, index
    %12 = arith.index_cast %offset_2 : index to i64
    %c8_i64_5 = arith.constant 8 : i64
    %13 = arith.muli %12, %c8_i64_5 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_6 = memref.extract_aligned_pointer_as_index %arg2 : memref<?xf64> -> index
    %16 = arith.index_cast %intptr_6 : index to i64
    %base_buffer_7, %offset_8, %sizes_9, %strides_10 = memref.extract_strided_metadata %arg2 : memref<?xf64> -> memref<f64>, index, index, index
    %17 = arith.index_cast %offset_8 : index to i64
    %c8_i64_11 = arith.constant 8 : i64
    %18 = arith.muli %17, %c8_i64_11 : i64
    %19 = arith.addi %16, %18 : i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    call @polygeist_cublas_dtrsv_lower_row_major(%5, %10, %15, %20) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cublas_dtrsv_lower_row_major(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
}

