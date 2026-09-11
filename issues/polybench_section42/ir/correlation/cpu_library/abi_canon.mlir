module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %0 = arith.index_cast %dim : index to i32
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %1 = arith.index_cast %dim_0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?xf64> -> index
    %2 = arith.index_cast %intptr : index to i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %4 = arith.index_cast %strides#0 : index to i32
    %intptr_1 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %5 = arith.index_cast %intptr_1 : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %7 = arith.index_cast %strides_5#0 : index to i32
    %intptr_6 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %8 = arith.index_cast %intptr_6 : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %intptr_7 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %10 = arith.index_cast %intptr_7 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    call @polygeist_cublas_dcorrelation_row_major(%0, %1, %arg2, %3, %4, %6, %7, %9, %11) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cublas_dcorrelation_row_major(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr)
}

