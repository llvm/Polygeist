module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sum(%arg0: memref<?x64xf64>, %arg1: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %cast = memref.cast %arg0 : memref<?x64xf64> to memref<?x?xf64>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = memref.dim %cast, %c0 : memref<?x?xf64>
    %2 = arith.index_cast %dim : index to i32
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %cast, %c1 : memref<?x?xf64>
    %3 = arith.index_cast %dim_0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %cast : memref<?x?xf64> -> index
    %4 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %cast : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %5 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %6 = arith.muli %5, %c8_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %intptr_1 = memref.extract_aligned_pointer_as_index %arg1 : memref<?xf64> -> index
    %9 = arith.index_cast %intptr_1 : index to i64
    %base_buffer_2, %offset_3, %sizes_4, %strides_5 = memref.extract_strided_metadata %arg1 : memref<?xf64> -> memref<f64>, index, index, index
    %10 = arith.index_cast %offset_3 : index to i64
    %c8_i64_6 = arith.constant 8 : i64
    %11 = arith.muli %10, %c8_i64_6 : i64
    %12 = arith.addi %9, %11 : i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cub_segmented_reduce_f64(%c0_i32, %2, %3, %8, %13) : (i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cub_segmented_reduce_f64(i32, i32, i32, !llvm.ptr, !llvm.ptr)
}
