module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_argmax_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c63 = arith.constant 63 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca(%c32) : memref<?xi32>
    %alloca_0 = memref.alloca(%c32) : memref<?xf32>
    %c0_i32_1 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x64xf32>
    %0 = arith.index_cast %dim : index to i32
    %c64_i32 = arith.constant 64 : i32
    %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<?x64xf32> -> index
    %1 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<?x64xf32> -> memref<f32>, index, index, index, index, index
    %2 = arith.index_cast %offset : index to i64
    %c4_i64 = arith.constant 4 : i64
    %3 = arith.muli %2, %c4_i64 : i64
    %4 = arith.addi %1, %3 : i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg1 : memref<?xi32> -> index
    %6 = arith.index_cast %intptr_2 : index to i64
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg1 : memref<?xi32> -> memref<i32>, index, index, index
    %7 = arith.index_cast %offset_4 : index to i64
    %c4_i64_7 = arith.constant 4 : i64
    %8 = arith.muli %7, %c4_i64_7 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    call @polygeist_cub_segmented_argreduce_f32(%c0_i32_1, %0, %c64_i32, %5, %10) : (i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cub_segmented_argreduce_f32(i32, i32, i32, !llvm.ptr, !llvm.ptr)
}
