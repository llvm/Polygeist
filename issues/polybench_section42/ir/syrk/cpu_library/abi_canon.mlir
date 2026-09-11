module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8_i64 = arith.constant 8 : i64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %subview = memref.subview %arg5[0, 0] [%1, %0] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %2 = arith.index_cast %strides#0 : index to i32
    %3 = arith.index_cast %strides_3#0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %4 = arith.index_cast %intptr : index to i64
    %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %5 = arith.index_cast %offset_5 : index to i64
    %6 = arith.muli %5, %c8_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %intptr_8 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %9 = arith.index_cast %intptr_8 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    call @polygeist_cublas_dsyrk_lower(%arg0, %arg1, %arg2, %8, %2, %arg3, %10, %3) : (i32, i32, f64, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    return
  }
  func.func private @polygeist_cublas_dsyrk_lower(i32, i32, f64, !llvm.ptr, i32, f64, !llvm.ptr, i32)
}

