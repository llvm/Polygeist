module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_bilinear_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x16x20xf32>, %arg2: memref<?x20xf32>, %arg3: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c20 = arith.constant 20 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x16x20xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x20xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?x24xf32>
    %extracted_slice = tensor.extract_slice %3[0, 0] [%c8, %c24] [1, 1] : tensor<?x24xf32> to tensor<?x?xf32>
    %cast = memref.cast %arg0 : memref<?x16xf32> to memref<?x?xf32>
    %cast_0 = memref.cast %arg1 : memref<?x16x20xf32> to memref<?x?x?xf32>
    %cast_1 = memref.cast %arg2 : memref<?x20xf32> to memref<?x?xf32>
    %cast_2 = memref.cast %arg3 : memref<?x24xf32> to memref<?x?xf32>
    %c0_i64 = arith.constant 0 : i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %cast : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %4 = arith.index_cast %strides#0 : index to i64
    %5 = arith.index_cast %sizes#0 : index to i64
    %6 = arith.index_cast %strides#1 : index to i64
    %7 = arith.index_cast %sizes#1 : index to i64
    %c0_i64_3 = arith.constant 0 : i64
    %base_buffer_4, %offset_5, %sizes_6:3, %strides_7:3 = memref.extract_strided_metadata %cast_0 : memref<?x?x?xf32> -> memref<f32>, index, index, index, index, index, index, index
    %8 = arith.index_cast %strides_7#0 : index to i64
    %9 = arith.index_cast %sizes_6#0 : index to i64
    %10 = arith.index_cast %strides_7#1 : index to i64
    %11 = arith.index_cast %sizes_6#1 : index to i64
    %12 = arith.index_cast %strides_7#2 : index to i64
    %13 = arith.index_cast %sizes_6#2 : index to i64
    %c0_i64_8 = arith.constant 0 : i64
    %base_buffer_9, %offset_10, %sizes_11:2, %strides_12:2 = memref.extract_strided_metadata %cast_1 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %14 = arith.index_cast %strides_12#0 : index to i64
    %15 = arith.index_cast %sizes_11#0 : index to i64
    %16 = arith.index_cast %strides_12#1 : index to i64
    %17 = arith.index_cast %sizes_11#1 : index to i64
    %c0_i64_13 = arith.constant 0 : i64
    %base_buffer_14, %offset_15, %sizes_16:2, %strides_17:2 = memref.extract_strided_metadata %cast_2 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %18 = arith.index_cast %strides_17#0 : index to i64
    %19 = arith.index_cast %sizes_16#0 : index to i64
    %20 = arith.index_cast %strides_17#1 : index to i64
    %21 = arith.index_cast %sizes_16#1 : index to i64
    %alloca = memref.alloca() : memref<34xi64>
    %alloca_18 = memref.alloca() : memref<4xi64>
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    memref.store %c1_i64, %alloca[%c0] : memref<34xi64>
    %c3_i64 = arith.constant 3 : i64
    %c1 = arith.constant 1 : index
    memref.store %c3_i64, %alloca[%c1] : memref<34xi64>
    %c0_i64_19 = arith.constant 0 : i64
    %c2 = arith.constant 2 : index
    memref.store %c0_i64_19, %alloca[%c2] : memref<34xi64>
    %c2_i64 = arith.constant 2 : i64
    %c3 = arith.constant 3 : index
    memref.store %c2_i64, %alloca[%c3] : memref<34xi64>
    %c7 = arith.constant 7 : index
    memref.store %5, %alloca[%c7] : memref<34xi64>
    %c8_20 = arith.constant 8 : index
    memref.store %4, %alloca[%c8_20] : memref<34xi64>
    %c0_i64_21 = arith.constant 0 : i64
    %c9 = arith.constant 9 : index
    memref.store %c0_i64_21, %alloca[%c9] : memref<34xi64>
    %c10 = arith.constant 10 : index
    memref.store %7, %alloca[%c10] : memref<34xi64>
    %c11 = arith.constant 11 : index
    memref.store %6, %alloca[%c11] : memref<34xi64>
    %c2_i64_22 = arith.constant 2 : i64
    %c12 = arith.constant 12 : index
    memref.store %c2_i64_22, %alloca[%c12] : memref<34xi64>
    %intptr = memref.extract_aligned_pointer_as_index %cast : memref<?x?xf32> -> index
    %22 = arith.index_cast %intptr : index to i64
    %base_buffer_23, %offset_24, %sizes_25:2, %strides_26:2 = memref.extract_strided_metadata %cast : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %23 = arith.index_cast %offset_24 : index to i64
    %c4_i64 = arith.constant 4 : i64
    %24 = arith.muli %23, %c4_i64 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %c0_27 = arith.constant 0 : index
    memref.store %27, %alloca_18[%c0_27] : memref<4xi64>
    %c3_i64_28 = arith.constant 3 : i64
    %c4 = arith.constant 4 : index
    memref.store %c3_i64_28, %alloca[%c4] : memref<34xi64>
    %c13 = arith.constant 13 : index
    memref.store %9, %alloca[%c13] : memref<34xi64>
    %c14 = arith.constant 14 : index
    memref.store %8, %alloca[%c14] : memref<34xi64>
    %c1_i64_29 = arith.constant 1 : i64
    %c15 = arith.constant 15 : index
    memref.store %c1_i64_29, %alloca[%c15] : memref<34xi64>
    %c16_30 = arith.constant 16 : index
    memref.store %11, %alloca[%c16_30] : memref<34xi64>
    %c17 = arith.constant 17 : index
    memref.store %10, %alloca[%c17] : memref<34xi64>
    %c2_i64_31 = arith.constant 2 : i64
    %c18 = arith.constant 18 : index
    memref.store %c2_i64_31, %alloca[%c18] : memref<34xi64>
    %c19 = arith.constant 19 : index
    memref.store %13, %alloca[%c19] : memref<34xi64>
    %c20_32 = arith.constant 20 : index
    memref.store %12, %alloca[%c20_32] : memref<34xi64>
    %c3_i64_33 = arith.constant 3 : i64
    %c21 = arith.constant 21 : index
    memref.store %c3_i64_33, %alloca[%c21] : memref<34xi64>
    %intptr_34 = memref.extract_aligned_pointer_as_index %cast_0 : memref<?x?x?xf32> -> index
    %28 = arith.index_cast %intptr_34 : index to i64
    %base_buffer_35, %offset_36, %sizes_37:3, %strides_38:3 = memref.extract_strided_metadata %cast_0 : memref<?x?x?xf32> -> memref<f32>, index, index, index, index, index, index, index
    %29 = arith.index_cast %offset_36 : index to i64
    %c4_i64_39 = arith.constant 4 : i64
    %30 = arith.muli %29, %c4_i64_39 : i64
    %31 = arith.addi %28, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %c1_40 = arith.constant 1 : index
    memref.store %33, %alloca_18[%c1_40] : memref<4xi64>
    %c2_i64_41 = arith.constant 2 : i64
    %c5 = arith.constant 5 : index
    memref.store %c2_i64_41, %alloca[%c5] : memref<34xi64>
    %c22 = arith.constant 22 : index
    memref.store %15, %alloca[%c22] : memref<34xi64>
    %c23 = arith.constant 23 : index
    memref.store %14, %alloca[%c23] : memref<34xi64>
    %c0_i64_42 = arith.constant 0 : i64
    %c24_43 = arith.constant 24 : index
    memref.store %c0_i64_42, %alloca[%c24_43] : memref<34xi64>
    %c25 = arith.constant 25 : index
    memref.store %17, %alloca[%c25] : memref<34xi64>
    %c26 = arith.constant 26 : index
    memref.store %16, %alloca[%c26] : memref<34xi64>
    %c3_i64_44 = arith.constant 3 : i64
    %c27 = arith.constant 27 : index
    memref.store %c3_i64_44, %alloca[%c27] : memref<34xi64>
    %intptr_45 = memref.extract_aligned_pointer_as_index %cast_1 : memref<?x?xf32> -> index
    %34 = arith.index_cast %intptr_45 : index to i64
    %base_buffer_46, %offset_47, %sizes_48:2, %strides_49:2 = memref.extract_strided_metadata %cast_1 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %35 = arith.index_cast %offset_47 : index to i64
    %c4_i64_50 = arith.constant 4 : i64
    %36 = arith.muli %35, %c4_i64_50 : i64
    %37 = arith.addi %34, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %c2_51 = arith.constant 2 : index
    memref.store %39, %alloca_18[%c2_51] : memref<4xi64>
    %c2_i64_52 = arith.constant 2 : i64
    %c6 = arith.constant 6 : index
    memref.store %c2_i64_52, %alloca[%c6] : memref<34xi64>
    %c28 = arith.constant 28 : index
    memref.store %19, %alloca[%c28] : memref<34xi64>
    %c29 = arith.constant 29 : index
    memref.store %18, %alloca[%c29] : memref<34xi64>
    %c0_i64_53 = arith.constant 0 : i64
    %c30 = arith.constant 30 : index
    memref.store %c0_i64_53, %alloca[%c30] : memref<34xi64>
    %c31 = arith.constant 31 : index
    memref.store %21, %alloca[%c31] : memref<34xi64>
    %c32 = arith.constant 32 : index
    memref.store %20, %alloca[%c32] : memref<34xi64>
    %c1_i64_54 = arith.constant 1 : i64
    %c33 = arith.constant 33 : index
    memref.store %c1_i64_54, %alloca[%c33] : memref<34xi64>
    %intptr_55 = memref.extract_aligned_pointer_as_index %cast_2 : memref<?x?xf32> -> index
    %40 = arith.index_cast %intptr_55 : index to i64
    %base_buffer_56, %offset_57, %sizes_58:2, %strides_59:2 = memref.extract_strided_metadata %cast_2 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    %41 = arith.index_cast %offset_57 : index to i64
    %c4_i64_60 = arith.constant 4 : i64
    %42 = arith.muli %41, %c4_i64_60 : i64
    %43 = arith.addi %40, %42 : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %c3_61 = arith.constant 3 : index
    memref.store %45, %alloca_18[%c3_61] : memref<4xi64>
    %intptr_62 = memref.extract_aligned_pointer_as_index %alloca_18 : memref<4xi64> -> index
    %46 = arith.index_cast %intptr_62 : index to i64
    %base_buffer_63, %offset_64, %sizes_65, %strides_66 = memref.extract_strided_metadata %alloca_18 : memref<4xi64> -> memref<i64>, index, index, index
    %47 = arith.index_cast %offset_64 : index to i64
    %c8_i64 = arith.constant 8 : i64
    %48 = arith.muli %47, %c8_i64 : i64
    %49 = arith.addi %46, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %intptr_67 = memref.extract_aligned_pointer_as_index %alloca : memref<34xi64> -> index
    %51 = arith.index_cast %intptr_67 : index to i64
    %base_buffer_68, %offset_69, %sizes_70, %strides_71 = memref.extract_strided_metadata %alloca : memref<34xi64> -> memref<i64>, index, index, index
    %52 = arith.index_cast %offset_69 : index to i64
    %c8_i64_72 = arith.constant 8 : i64
    %53 = arith.muli %52, %c8_i64_72 : i64
    %54 = arith.addi %51, %53 : i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    call @polygeist_cutensornet_network_f32(%50, %55) : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cutensornet_network_f32(!llvm.ptr, !llvm.ptr)
}
