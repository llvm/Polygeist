module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_allany_dims_cpu(%arg0: memref<?x64xi32>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg1, %c0_i32 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %1 = arith.cmpi ne, %arg1, %c0_i32_1 : i32
    %c1_i32 = arith.constant 1 : i32
    %2 = arith.select %1, %c0_i32_0, %c1_i32 : i32
    %3 = arith.select %1, %arg0, %arg0 : memref<?x64xi32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %3, %c0 : memref<?x64xi32>
    %4 = arith.index_cast %dim : index to i32
    %c64_i32 = arith.constant 64 : i32
    %intptr = memref.extract_aligned_pointer_as_index %3 : memref<?x64xi32> -> index
    %5 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %3 : memref<?x64xi32> -> memref<i32>, index, index, index, index, index
    %6 = arith.index_cast %offset : index to i64
    %c4_i64 = arith.constant 4 : i64
    %7 = arith.muli %6, %c4_i64 : i64
    %8 = arith.addi %5, %7 : i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg2 : memref<?xi32> -> index
    %10 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg2 : memref<?xi32> -> memref<i32>, index, index, index
    %11 = arith.index_cast %offset_4 : index to i64
    %c4_i64_7 = arith.constant 4 : i64
    %12 = arith.muli %11, %c4_i64_7 : i64
    %13 = arith.addi %10, %12 : i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    call @polygeist_cub_segmented_reduce_i32(%2, %4, %c64_i32, %9, %14) : (i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cub_segmented_reduce_i32(i32, i32, i32, !llvm.ptr, !llvm.ptr)
}
