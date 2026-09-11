#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @kernel_3mm_kernel94865853255664(%arg0: index, %arg1: index, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %12 = affine.apply #map(%0)[%arg0, %arg1]
      %13 = affine.apply #map(%1)[%arg0, %arg1]
      %14 = memref.load %arg2[%12, %13] : memref<?x?xf64, strided<[?, 1], offset: ?>>
      memref.store %14, %arg3[%12, %13] : memref<?x?xf64, strided<[?, 1], offset: ?>>
      gpu.return
    }
    gpu.func @kernel_3mm_kernel94865853250560(%arg0: index, %arg1: index, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %12 = affine.apply #map(%0)[%arg0, %arg1]
      %13 = affine.apply #map(%1)[%arg0, %arg1]
      %14 = memref.load %arg2[%12, %13] : memref<?x?xf64, strided<[?, 1], offset: ?>>
      memref.store %14, %arg3[%12, %13] : memref<?x?xf64, strided<[?, 1], offset: ?>>
      gpu.return
    }
    gpu.func @kernel_3mm_kernel94865853242896(%arg0: index, %arg1: index, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %12 = affine.apply #map(%0)[%arg0, %arg1]
      %13 = affine.apply #map(%1)[%arg0, %arg1]
      %14 = memref.load %arg2[%12, %13] : memref<?x?xf64, strided<[?, 1], offset: ?>>
      memref.store %14, %arg3[%12, %13] : memref<?x?xf64, strided<[?, 1], offset: ?>>
      gpu.return
    }
  }
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_residual_pipeline, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = -1890640588580836432 : i64, polygeist.pipeline_scope_coalesced} {
    %c-1890640588580836432_i64 = arith.constant -1890640588580836432 : i64
    call @polygeist_gpu_region_timing_begin(%c-1890640588580836432_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg5, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg5, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg5 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg6, %c0_1 : memref<?x?xf64>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg6, %c1_3 : memref<?x?xf64>
    %c2_i32_5 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_5) : (i32) -> ()
    %memref_6, %asyncToken_7 = gpu.alloc async [%1] (%dim_2, %dim_4) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_8 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_8) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_7] %memref_6, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg7, %c0_9 : memref<?x?xf64>
    %c1_11 = arith.constant 1 : index
    %dim_12 = memref.dim %arg7, %c1_11 : memref<?x?xf64>
    %c2_i32_13 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_13) : (i32) -> ()
    %memref_14, %asyncToken_15 = gpu.alloc async [%2] (%dim_10, %dim_12) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_16 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_16) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_15] %memref_14, %arg7 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_17 = arith.constant 0 : index
    %dim_18 = memref.dim %arg8, %c0_17 : memref<?x?xf64>
    %c1_19 = arith.constant 1 : index
    %dim_20 = memref.dim %arg8, %c1_19 : memref<?x?xf64>
    %c2_i32_21 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_21) : (i32) -> ()
    %memref_22, %asyncToken_23 = gpu.alloc async [%3] (%dim_18, %dim_20) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_24 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_24) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_23] %memref_22, %arg8 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg9, %c0_25 : memref<?x?xf64>
    %c1_27 = arith.constant 1 : index
    %dim_28 = memref.dim %arg9, %c1_27 : memref<?x?xf64>
    %c2_i32_29 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_29) : (i32) -> ()
    %memref_30, %asyncToken_31 = gpu.alloc async [%4] (%dim_26, %dim_28) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_32) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_31] %memref_30, %arg9 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_33 = arith.constant 0 : index
    %dim_34 = memref.dim %arg10, %c0_33 : memref<?x?xf64>
    %c1_35 = arith.constant 1 : index
    %dim_36 = memref.dim %arg10, %c1_35 : memref<?x?xf64>
    %c2_i32_37 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_37) : (i32) -> ()
    %memref_38, %asyncToken_39 = gpu.alloc async [%5] (%dim_34, %dim_36) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_40 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_40) : (i32) -> ()
    %6 = gpu.memcpy async [%asyncToken_39] %memref_38, %arg10 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_41 = arith.constant 0 : index
    %dim_42 = memref.dim %arg11, %c0_41 : memref<?x?xf64>
    %c1_43 = arith.constant 1 : index
    %dim_44 = memref.dim %arg11, %c1_43 : memref<?x?xf64>
    %c2_i32_45 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_45) : (i32) -> ()
    %memref_46, %asyncToken_47 = gpu.alloc async [%6] (%dim_42, %dim_44) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_48 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_48) : (i32) -> ()
    %7 = gpu.memcpy async [%asyncToken_47] %memref_46, %arg11 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%7]
    %cast = memref.cast %memref : memref<?x?xf64> to memref<*xf64>
    %cast_49 = memref.cast %memref_6 : memref<?x?xf64> to memref<*xf64>
    %cast_50 = memref.cast %memref_14 : memref<?x?xf64> to memref<*xf64>
    %cast_51 = memref.cast %memref_22 : memref<?x?xf64> to memref<*xf64>
    %cast_52 = memref.cast %memref_30 : memref<?x?xf64> to memref<*xf64>
    %cast_53 = memref.cast %memref_38 : memref<?x?xf64> to memref<*xf64>
    %cast_54 = memref.cast %memref_46 : memref<?x?xf64> to memref<*xf64>
    %c0_55 = arith.constant 0 : index
    %c1_56 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg2 : i32 to index
    %10 = arith.index_cast %arg4 : i32 to index
    %11 = arith.index_cast %arg3 : i32 to index
    %12 = arith.index_cast %arg0 : i32 to index
    %dim_57 = memref.dim %memref, %c0_55 : memref<?x?xf64>
    %13 = arith.index_cast %dim_57 : index to i32
    %dim_58 = memref.dim %memref, %c1_56 : memref<?x?xf64>
    %14 = arith.index_cast %dim_58 : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %memref : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_2d(%13, %14, %16, %14) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview = memref.subview %memref_6[0, 0] [%12, %9] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_59 = memref.subview %memref_14[0, 0] [%9, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_60 = memref.subview %memref[0, 0] [%12, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_61 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %17 = arith.index_cast %intptr_61 : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %18 = arith.index_cast %offset : index to i64
    %19 = arith.muli %18, %c8_i64 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %intptr_62 = memref.extract_aligned_pointer_as_index %subview_59 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %22 = arith.index_cast %intptr_62 : index to i64
    %base_buffer_63, %offset_64, %sizes_65:2, %strides_66:2 = memref.extract_strided_metadata %subview_59 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %23 = arith.index_cast %offset_64 : index to i64
    %24 = arith.muli %23, %c8_i64 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %intptr_67 = memref.extract_aligned_pointer_as_index %subview_60 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %27 = arith.index_cast %intptr_67 : index to i64
    %base_buffer_68, %offset_69, %sizes_70:2, %strides_71:2 = memref.extract_strided_metadata %subview_60 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %28 = arith.index_cast %offset_69 : index to i64
    %29 = arith.muli %28, %c8_i64 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg1, %arg2, %cst, %21, %arg2, %26, %arg1, %cst, %31, %arg1) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %subview_72 = memref.subview %memref[0, 0] [%12, %8] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c1_73 = arith.constant 1 : index
    %32 = affine.apply #map1(%12)[%c0_55, %c1_56]
    %33 = affine.apply #map1(%8)[%c0_55, %c1_56]
    %c0_i64 = arith.constant 0 : i64
    %34 = call @polygeist_cuda_graph_begin(%c0_i64) : (i64) -> i32
    %c0_i32 = arith.constant 0 : i32
    %35 = arith.cmpi ne, %34, %c0_i32 : i32
    scf.if %35 {
      gpu.launch_func  @__polygeist_gpu_module::@kernel_3mm_kernel94865853255664 blocks in (%32, %33, %c1_73) threads in (%c1_73, %c1_73, %c1_73)  args(%c1_56 : index, %c0_55 : index, %subview_60 : memref<?x?xf64, strided<[?, 1], offset: ?>>, %subview_72 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {polygeist.cuda_graph_safe}
      func.call @polygeist_cuda_graph_end(%c0_i64) : (i64) -> ()
    } {polygeist.cuda_graph_scope}
    %dim_74 = memref.dim %memref_22, %c0_55 : memref<?x?xf64>
    %36 = arith.index_cast %dim_74 : index to i32
    %dim_75 = memref.dim %memref_22, %c1_56 : memref<?x?xf64>
    %37 = arith.index_cast %dim_75 : index to i32
    %intptr_76 = memref.extract_aligned_pointer_as_index %memref_22 : memref<?x?xf64> -> index
    %38 = arith.index_cast %intptr_76 : index to i64
    %39 = llvm.inttoptr %38 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%36, %37, %39, %37) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview_77 = memref.subview %memref_30[0, 0] [%8, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_78 = memref.subview %memref_38[0, 0] [%10, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_79 = memref.subview %memref_22[0, 0] [%8, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_80 = memref.extract_aligned_pointer_as_index %subview_77 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %40 = arith.index_cast %intptr_80 : index to i64
    %base_buffer_81, %offset_82, %sizes_83:2, %strides_84:2 = memref.extract_strided_metadata %subview_77 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %41 = arith.index_cast %offset_82 : index to i64
    %42 = arith.muli %41, %c8_i64 : i64
    %43 = arith.addi %40, %42 : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
    %intptr_85 = memref.extract_aligned_pointer_as_index %subview_78 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %45 = arith.index_cast %intptr_85 : index to i64
    %base_buffer_86, %offset_87, %sizes_88:2, %strides_89:2 = memref.extract_strided_metadata %subview_78 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %46 = arith.index_cast %offset_87 : index to i64
    %47 = arith.muli %46, %c8_i64 : i64
    %48 = arith.addi %45, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %intptr_90 = memref.extract_aligned_pointer_as_index %subview_79 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %50 = arith.index_cast %intptr_90 : index to i64
    %base_buffer_91, %offset_92, %sizes_93:2, %strides_94:2 = memref.extract_strided_metadata %subview_79 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %51 = arith.index_cast %offset_92 : index to i64
    %52 = arith.muli %51, %c8_i64 : i64
    %53 = arith.addi %50, %52 : i64
    %54 = llvm.inttoptr %53 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg1, %arg3, %arg4, %cst, %44, %arg4, %49, %arg3, %cst, %54, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %subview_95 = memref.subview %memref_22[0, 0] [%8, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c1_96 = arith.constant 1 : index
    %55 = affine.apply #map1(%8)[%c0_55, %c1_56]
    %56 = affine.apply #map1(%11)[%c0_55, %c1_56]
    %c1_i64 = arith.constant 1 : i64
    %57 = call @polygeist_cuda_graph_begin(%c1_i64) : (i64) -> i32
    %c0_i32_97 = arith.constant 0 : i32
    %58 = arith.cmpi ne, %57, %c0_i32_97 : i32
    scf.if %58 {
      gpu.launch_func  @__polygeist_gpu_module::@kernel_3mm_kernel94865853250560 blocks in (%55, %56, %c1_96) threads in (%c1_96, %c1_96, %c1_96)  args(%c1_56 : index, %c0_55 : index, %subview_79 : memref<?x?xf64, strided<[?, 1], offset: ?>>, %subview_95 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {polygeist.cuda_graph_safe}
      func.call @polygeist_cuda_graph_end(%c1_i64) : (i64) -> ()
    } {polygeist.cuda_graph_scope}
    %dim_98 = memref.dim %memref_46, %c0_55 : memref<?x?xf64>
    %59 = arith.index_cast %dim_98 : index to i32
    %dim_99 = memref.dim %memref_46, %c1_56 : memref<?x?xf64>
    %60 = arith.index_cast %dim_99 : index to i32
    %intptr_100 = memref.extract_aligned_pointer_as_index %memref_46 : memref<?x?xf64> -> index
    %61 = arith.index_cast %intptr_100 : index to i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_2d(%59, %60, %62, %60) : (i32, i32, !llvm.ptr, i32) -> ()
    %subview_101 = memref.subview %memref_46[0, 0] [%12, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %intptr_102 = memref.extract_aligned_pointer_as_index %subview_60 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %63 = arith.index_cast %intptr_102 : index to i64
    %base_buffer_103, %offset_104, %sizes_105:2, %strides_106:2 = memref.extract_strided_metadata %subview_60 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %64 = arith.index_cast %offset_104 : index to i64
    %65 = arith.muli %64, %c8_i64 : i64
    %66 = arith.addi %63, %65 : i64
    %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
    %intptr_107 = memref.extract_aligned_pointer_as_index %subview_79 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %68 = arith.index_cast %intptr_107 : index to i64
    %base_buffer_108, %offset_109, %sizes_110:2, %strides_111:2 = memref.extract_strided_metadata %subview_79 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %69 = arith.index_cast %offset_109 : index to i64
    %70 = arith.muli %69, %c8_i64 : i64
    %71 = arith.addi %68, %70 : i64
    %72 = llvm.inttoptr %71 : i64 to !llvm.ptr
    %intptr_112 = memref.extract_aligned_pointer_as_index %subview_101 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %73 = arith.index_cast %intptr_112 : index to i64
    %base_buffer_113, %offset_114, %sizes_115:2, %strides_116:2 = memref.extract_strided_metadata %subview_101 : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %74 = arith.index_cast %offset_114 : index to i64
    %75 = arith.muli %74, %c8_i64 : i64
    %76 = arith.addi %73, %75 : i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemm(%arg0, %arg3, %arg1, %cst, %67, %arg1, %72, %arg3, %cst, %77, %arg3) : (i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32) -> ()
    %subview_117 = memref.subview %memref_46[0, 0] [%12, %11] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c1_118 = arith.constant 1 : index
    %78 = affine.apply #map1(%12)[%c0_55, %c1_56]
    %79 = affine.apply #map1(%11)[%c0_55, %c1_56]
    %c2_i64 = arith.constant 2 : i64
    %80 = call @polygeist_cuda_graph_begin(%c2_i64) : (i64) -> i32
    %c0_i32_119 = arith.constant 0 : i32
    %81 = arith.cmpi ne, %80, %c0_i32_119 : i32
    scf.if %81 {
      gpu.launch_func  @__polygeist_gpu_module::@kernel_3mm_kernel94865853242896 blocks in (%78, %79, %c1_118) threads in (%c1_118, %c1_118, %c1_118)  args(%c1_56 : index, %c0_55 : index, %subview_101 : memref<?x?xf64, strided<[?, 1], offset: ?>>, %subview_117 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {polygeist.cuda_graph_safe}
      func.call @polygeist_cuda_graph_end(%c2_i64) : (i64) -> ()
    } {polygeist.cuda_graph_scope}
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %82 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %83 = gpu.memcpy async [%82] %arg5, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %84 = gpu.dealloc async [%83] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_120 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_120) : (i32) -> ()
    %85 = gpu.memcpy async [%84] %arg6, %memref_6 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_121 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_121) : (i32) -> ()
    %86 = gpu.dealloc async [%85] %memref_6 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_122 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_122) : (i32) -> ()
    %87 = gpu.memcpy async [%86] %arg7, %memref_14 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_123 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_123) : (i32) -> ()
    %88 = gpu.dealloc async [%87] %memref_14 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_124 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_124) : (i32) -> ()
    %89 = gpu.memcpy async [%88] %arg8, %memref_22 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_125 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_125) : (i32) -> ()
    %90 = gpu.dealloc async [%89] %memref_22 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_126 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_126) : (i32) -> ()
    %91 = gpu.memcpy async [%90] %arg9, %memref_30 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_127 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_127) : (i32) -> ()
    %92 = gpu.dealloc async [%91] %memref_30 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_128 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_128) : (i32) -> ()
    %93 = gpu.memcpy async [%92] %arg10, %memref_38 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_129 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_129) : (i32) -> ()
    %94 = gpu.dealloc async [%93] %memref_38 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_130 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_130) : (i32) -> ()
    %95 = gpu.memcpy async [%94] %arg11, %memref_46 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_131 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_131) : (i32) -> ()
    %96 = gpu.dealloc async [%95] %memref_46 : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%96]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_memset_zero_2d(i32, i32, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemm(i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

