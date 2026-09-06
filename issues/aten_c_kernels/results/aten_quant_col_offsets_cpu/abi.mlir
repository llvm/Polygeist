module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_quant_col_offsets_cpu(%arg0: memref<?x48xi8>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64 = arith.constant 64 : index
    %c48 = arith.constant 48 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x48xi8>
    %1 = bufferization.to_tensor %arg2 : memref<?xi32>
    %2 = arith.muli %arg1, %c64_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %c48_i32 = arith.constant 48 : i32
    %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<?x48xi8> -> index
    %3 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<?x48xi8> -> memref<i8>, index, index, index, index, index
    %4 = arith.index_cast %offset : index to i64
    %c1_i64 = arith.constant 1 : i64
    %5 = arith.muli %4, %c1_i64 : i64
    %6 = arith.addi %3, %5 : i64
    %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %intptr_1 = memref.extract_aligned_pointer_as_index %arg2 : memref<?xi32> -> index
    %8 = arith.index_cast %intptr_1 : index to i64
    %base_buffer_2, %offset_3, %sizes_4, %strides_5 = memref.extract_strided_metadata %arg2 : memref<?xi32> -> memref<i32>, index, index, index
    %9 = arith.index_cast %offset_3 : index to i64
    %c4_i64 = arith.constant 4 : i64
    %10 = arith.muli %9, %c4_i64 : i64
    %11 = arith.addi %8, %10 : i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    call @polygeist_cub_quant_col_offsets_i8_i32(%c64_i32_0, %c48_i32, %2, %7, %12) : (i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cub_quant_col_offsets_i8_i32(i32, i32, i32, !llvm.ptr, !llvm.ptr)
}
