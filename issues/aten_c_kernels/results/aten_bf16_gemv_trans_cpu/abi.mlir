module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_bf16_gemv_trans_cpu(%arg0: memref<?x128xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cast = memref.cast %arg0 : memref<?x128xf32> to memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %cast, %c0 : memref<?x?xf32>
    %0 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %cast, %c1 : memref<?x?xf32>
    %1 = arith.index_cast %dim_0 : index to i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %intptr = memref.extract_aligned_pointer_as_index %cast : memref<?x?xf32> -> index
    %2 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %cast : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %3 = arith.index_cast %offset : index to i64
    %c4_i64 = arith.constant 4 : i64
    %4 = arith.muli %3, %c4_i64 : i64
    %5 = arith.addi %2, %4 : i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %arg1 : memref<?xf32> -> index
    %7 = arith.index_cast %intptr_3 : index to i64
    %base_buffer_4, %offset_5, %sizes_6, %strides_7 = memref.extract_strided_metadata %arg1 : memref<?xf32> -> memref<f32>, index, index, index
    %8 = arith.index_cast %offset_5 : index to i64
    %c4_i64_8 = arith.constant 4 : i64
    %9 = arith.muli %8, %c4_i64_8 : i64
    %10 = arith.addi %7, %9 : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %arg2 : memref<?xf32> -> index
    %12 = arith.index_cast %intptr_9 : index to i64
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %arg2 : memref<?xf32> -> memref<f32>, index, index, index
    %13 = arith.index_cast %offset_11 : index to i64
    %c4_i64_14 = arith.constant 4 : i64
    %14 = arith.muli %13, %c4_i64_14 : i64
    %15 = arith.addi %12, %14 : i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    call @polygeist_cublas_sgemv_T(%0, %1, %cst_1, %6, %1, %11, %cst_2, %16) : (i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, f32, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cublas_sgemv_T(i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, f32, !llvm.ptr)
}
