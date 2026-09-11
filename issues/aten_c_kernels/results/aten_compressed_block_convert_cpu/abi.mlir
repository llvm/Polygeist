module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_compressed_block_convert_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x16x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c-1 = arith.constant -1 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 4, 16, 4], strides: [256, 64, 4, 1] : memref<?x64xf32> to memref<16x4x16x4xf32, strided<[256, 64, 4, 1]>>
    %0 = bufferization.to_tensor %reinterpret_cast restrict : memref<16x4x16x4xf32, strided<[256, 64, 4, 1]>>
    %cast = tensor.cast %0 : tensor<16x4x16x4xf32> to tensor<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<?x16x4x4xf32>
    %cast_0 = tensor.cast %1 : tensor<?x16x4x4xf32> to tensor<?x?x?x?xf32>
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64_1 = arith.constant 16 : i64
    %c4_i64_2 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %2 = arith.muli %c1_i64, %c4_i64_2 : i64
    %3 = arith.muli %2, %c16_i64_1 : i64
    %4 = arith.muli %3, %c4_i64 : i64
    %5 = arith.muli %4, %c16_i64 : i64
    %c0_i64_3 = arith.constant 0 : i64
    %c0_4 = arith.constant 0 : index
    %dim = tensor.dim %1, %c0_4 : tensor<?x16x4x4xf32>
    %6 = arith.index_cast %dim : index to i64
    %c16_i64_5 = arith.constant 16 : i64
    %c4_i64_6 = arith.constant 4 : i64
    %c4_i64_7 = arith.constant 4 : i64
    %c1_i64_8 = arith.constant 1 : i64
    %7 = arith.muli %c1_i64_8, %c4_i64_7 : i64
    %8 = arith.muli %7, %c4_i64_6 : i64
    %9 = arith.muli %8, %c16_i64_5 : i64
    %10 = arith.muli %9, %6 : i64
    %alloca = memref.alloca() : memref<4xi64>
    %alloca_9 = memref.alloca() : memref<4xi64>
    %alloca_10 = memref.alloca() : memref<4xi32>
    %alloca_11 = memref.alloca() : memref<4xi64>
    %alloca_12 = memref.alloca() : memref<4xi64>
    %alloca_13 = memref.alloca() : memref<4xi32>
    %c0_14 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_15 = arith.constant 0 : i32
    memref.store %c16_i64, %alloca[%c0_14] : memref<4xi64>
    memref.store %4, %alloca_9[%c0_14] : memref<4xi64>
    memref.store %c0_i32, %alloca_10[%c0_14] : memref<4xi32>
    memref.store %6, %alloca_11[%c0_14] : memref<4xi64>
    memref.store %9, %alloca_12[%c0_14] : memref<4xi64>
    memref.store %c0_i32_15, %alloca_13[%c0_14] : memref<4xi32>
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    memref.store %c4_i64, %alloca[%c1] : memref<4xi64>
    memref.store %3, %alloca_9[%c1] : memref<4xi64>
    memref.store %c2_i32, %alloca_10[%c1] : memref<4xi32>
    memref.store %c16_i64_5, %alloca_11[%c1] : memref<4xi64>
    memref.store %8, %alloca_12[%c1] : memref<4xi64>
    memref.store %c1_i32, %alloca_13[%c1] : memref<4xi32>
    %c2 = arith.constant 2 : index
    %c1_i32_16 = arith.constant 1 : i32
    %c2_i32_17 = arith.constant 2 : i32
    memref.store %c16_i64_1, %alloca[%c2] : memref<4xi64>
    memref.store %2, %alloca_9[%c2] : memref<4xi64>
    memref.store %c1_i32_16, %alloca_10[%c2] : memref<4xi32>
    memref.store %c4_i64_6, %alloca_11[%c2] : memref<4xi64>
    memref.store %7, %alloca_12[%c2] : memref<4xi64>
    memref.store %c2_i32_17, %alloca_13[%c2] : memref<4xi32>
    %c3 = arith.constant 3 : index
    %c3_i32 = arith.constant 3 : i32
    %c3_i32_18 = arith.constant 3 : i32
    memref.store %c4_i64_2, %alloca[%c3] : memref<4xi64>
    memref.store %c1_i64, %alloca_9[%c3] : memref<4xi64>
    memref.store %c3_i32, %alloca_10[%c3] : memref<4xi32>
    memref.store %c4_i64_7, %alloca_11[%c3] : memref<4xi64>
    memref.store %c1_i64_8, %alloca_12[%c3] : memref<4xi64>
    memref.store %c3_i32_18, %alloca_13[%c3] : memref<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %intptr = memref.extract_aligned_pointer_as_index %reinterpret_cast : memref<16x4x16x4xf32, strided<[256, 64, 4, 1]>> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %reinterpret_cast : memref<16x4x16x4xf32, strided<[256, 64, 4, 1]>> -> memref<f32>, index, index, index, index, index, index, index, index, index
    %12 = arith.index_cast %offset : index to i64
    %c4_i64_19 = arith.constant 4 : i64
    %13 = arith.muli %12, %c4_i64_19 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_20 = memref.extract_aligned_pointer_as_index %arg1 : memref<?x16x4x4xf32> -> index
    %16 = arith.index_cast %intptr_20 : index to i64
    %base_buffer_21, %offset_22, %sizes_23:4, %strides_24:4 = memref.extract_strided_metadata %arg1 : memref<?x16x4x4xf32> -> memref<f32>, index, index, index, index, index, index, index, index, index
    %17 = arith.index_cast %offset_22 : index to i64
    %c4_i64_25 = arith.constant 4 : i64
    %18 = arith.muli %17, %c4_i64_25 : i64
    %19 = arith.addi %16, %18 : i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    %intptr_26 = memref.extract_aligned_pointer_as_index %alloca : memref<4xi64> -> index
    %21 = arith.index_cast %intptr_26 : index to i64
    %base_buffer_27, %offset_28, %sizes_29, %strides_30 = memref.extract_strided_metadata %alloca : memref<4xi64> -> memref<i64>, index, index, index
    %22 = arith.index_cast %offset_28 : index to i64
    %c8_i64 = arith.constant 8 : i64
    %23 = arith.muli %22, %c8_i64 : i64
    %24 = arith.addi %21, %23 : i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    %intptr_31 = memref.extract_aligned_pointer_as_index %alloca_9 : memref<4xi64> -> index
    %26 = arith.index_cast %intptr_31 : index to i64
    %base_buffer_32, %offset_33, %sizes_34, %strides_35 = memref.extract_strided_metadata %alloca_9 : memref<4xi64> -> memref<i64>, index, index, index
    %27 = arith.index_cast %offset_33 : index to i64
    %c8_i64_36 = arith.constant 8 : i64
    %28 = arith.muli %27, %c8_i64_36 : i64
    %29 = arith.addi %26, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %intptr_37 = memref.extract_aligned_pointer_as_index %alloca_10 : memref<4xi32> -> index
    %31 = arith.index_cast %intptr_37 : index to i64
    %base_buffer_38, %offset_39, %sizes_40, %strides_41 = memref.extract_strided_metadata %alloca_10 : memref<4xi32> -> memref<i32>, index, index, index
    %32 = arith.index_cast %offset_39 : index to i64
    %c4_i64_42 = arith.constant 4 : i64
    %33 = arith.muli %32, %c4_i64_42 : i64
    %34 = arith.addi %31, %33 : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %intptr_43 = memref.extract_aligned_pointer_as_index %alloca_11 : memref<4xi64> -> index
    %36 = arith.index_cast %intptr_43 : index to i64
    %base_buffer_44, %offset_45, %sizes_46, %strides_47 = memref.extract_strided_metadata %alloca_11 : memref<4xi64> -> memref<i64>, index, index, index
    %37 = arith.index_cast %offset_45 : index to i64
    %c8_i64_48 = arith.constant 8 : i64
    %38 = arith.muli %37, %c8_i64_48 : i64
    %39 = arith.addi %36, %38 : i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %intptr_49 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<4xi64> -> index
    %41 = arith.index_cast %intptr_49 : index to i64
    %base_buffer_50, %offset_51, %sizes_52, %strides_53 = memref.extract_strided_metadata %alloca_12 : memref<4xi64> -> memref<i64>, index, index, index
    %42 = arith.index_cast %offset_51 : index to i64
    %c8_i64_54 = arith.constant 8 : i64
    %43 = arith.muli %42, %c8_i64_54 : i64
    %44 = arith.addi %41, %43 : i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_55 = memref.extract_aligned_pointer_as_index %alloca_13 : memref<4xi32> -> index
    %46 = arith.index_cast %intptr_55 : index to i64
    %base_buffer_56, %offset_57, %sizes_58, %strides_59 = memref.extract_strided_metadata %alloca_13 : memref<4xi32> -> memref<i32>, index, index, index
    %47 = arith.index_cast %offset_57 : index to i64
    %c4_i64_60 = arith.constant 4 : i64
    %48 = arith.muli %47, %c4_i64_60 : i64
    %49 = arith.addi %46, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    call @polygeist_cutensor_permute_f32(%c4_i32, %25, %30, %35, %40, %45, %50, %15, %20) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %51 = bufferization.to_tensor %arg1 restrict writable : memref<?x16x4x4xf32>
    %cast_61 = tensor.cast %51 : tensor<?x16x4x4xf32> to tensor<?x?x?x?xf32>
    %cast_62 = tensor.cast %cast_61 : tensor<?x?x?x?xf32> to tensor<?x16x4x4xf32>
    %52 = bufferization.to_memref %cast_62 : memref<?x16x4x4xf32>
    memref.copy %52, %arg1 : memref<?x16x4x4xf32> to memref<?x16x4x4xf32>
    return
  }
  func.func private @polygeist_cutensor_permute_f32(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
}
