module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_diff_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c127 = arith.constant 127 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %c128_i32 = arith.constant 128 : i32
    %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<?xf32> -> index
    %2 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<?xf32> -> memref<f32>, index, index, index
    %3 = arith.index_cast %offset : index to i64
    %c4_i64 = arith.constant 4 : i64
    %4 = arith.muli %3, %c4_i64 : i64
    %5 = arith.addi %2, %4 : i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %intptr_0 = memref.extract_aligned_pointer_as_index %arg1 : memref<?xf32> -> index
    %7 = arith.index_cast %intptr_0 : index to i64
    %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %arg1 : memref<?xf32> -> memref<f32>, index, index, index
    %8 = arith.index_cast %offset_2 : index to i64
    %c4_i64_5 = arith.constant 4 : i64
    %9 = arith.muli %8, %c4_i64_5 : i64
    %10 = arith.addi %7, %9 : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    call @polygeist_cub_adjacent_difference_f32(%c128_i32, %6, %11) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cub_adjacent_difference_f32(i32, !llvm.ptr, !llvm.ptr)
}
