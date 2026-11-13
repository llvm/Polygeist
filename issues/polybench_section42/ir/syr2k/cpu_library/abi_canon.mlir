#map = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8_i64 = arith.constant 8 : i64
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = affine.apply #map()[%2]
    %subview = memref.subview %arg5[0, 0] [%3, %0] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_0 = memref.subview %arg6[0, 0] [%1, %0] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %4 = arith.index_cast %3 : index to i32
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg4 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %5 = arith.index_cast %strides#0 : index to i32
    %6 = arith.index_cast %strides_4#0 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %7 = arith.index_cast %intptr : index to i64
    %base_buffer_5, %offset_6, %sizes_7:2, %strides_8:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %8 = arith.index_cast %offset_6 : index to i64
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = arith.addi %7, %9 : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %12 = arith.index_cast %intptr_9 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %base_buffer_10, %offset_11, %sizes_12:2, %strides_13:2 = memref.extract_strided_metadata %subview_0 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %14 = arith.index_cast %strides_13#0 : index to i32
    %intptr_14 = memref.extract_aligned_pointer_as_index %subview_0 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %15 = arith.index_cast %intptr_14 : index to i64
    %base_buffer_15, %offset_16, %sizes_17:2, %strides_18:2 = memref.extract_strided_metadata %subview_0 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %16 = arith.index_cast %offset_16 : index to i64
    %17 = arith.muli %16, %c8_i64 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    call @polygeist_cublas_dsyr2k_lower(%4, %arg1, %arg2, %11, %5, %19, %14, %arg3, %13, %6) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    return
  }
  func.func private @polygeist_cublas_dsyr2k_lower(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
}

