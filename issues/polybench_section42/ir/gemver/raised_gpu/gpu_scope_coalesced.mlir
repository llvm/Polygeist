#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @kernel_gemver_kernel94051337448048(%arg0: index, %arg1: index, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
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
    gpu.func @kernel_gemver_kernel94051337811360(%arg0: index, %arg1: index, %arg2: memref<?xf64, strided<[1]>>, %arg3: memref<?xf64, strided<[1]>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
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
      %13 = memref.load %arg2[%12] : memref<?xf64, strided<[1]>>
      memref.store %13, %arg3[%12] : memref<?xf64, strided<[1]>>
      gpu.return
    }
    gpu.func @kernel_gemver_kernel94051337834192(%arg0: index, %arg1: index, %arg2: memref<?xf64, strided<[1]>>, %arg3: memref<?xf64, strided<[1]>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
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
      %13 = memref.load %arg2[%12] : memref<?xf64, strided<[1]>>
      memref.store %13, %arg3[%12] : memref<?xf64, strided<[1]>>
      gpu.return
    }
  }
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_residual_pipeline, polygeist.gpu_timing_instrumented, polygeist.gpu_timing_region_id = 6281486461887221929 : i64, polygeist.pipeline_scope_coalesced} {
    %c6281486461887221929_i64 = arith.constant 6281486461887221929 : i64
    call @polygeist_gpu_region_timing_begin(%c6281486461887221929_i64) : (i64) -> ()
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %c2_i32 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32) : (i32) -> ()
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32) : (i32) -> ()
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg3 : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg4, %c0_1 : memref<?xf64>
    %c2_i32_3 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_3) : (i32) -> ()
    %memref_4, %asyncToken_5 = gpu.alloc async [%1] (%dim_2) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_6 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_6) : (i32) -> ()
    %2 = gpu.memcpy async [%asyncToken_5] %memref_4, %arg4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg5, %c0_7 : memref<?xf64>
    %c2_i32_9 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_9) : (i32) -> ()
    %memref_10, %asyncToken_11 = gpu.alloc async [%2] (%dim_8) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_12 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_12) : (i32) -> ()
    %3 = gpu.memcpy async [%asyncToken_11] %memref_10, %arg5 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %arg6, %c0_13 : memref<?xf64>
    %c2_i32_15 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_15) : (i32) -> ()
    %memref_16, %asyncToken_17 = gpu.alloc async [%3] (%dim_14) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_18 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_18) : (i32) -> ()
    %4 = gpu.memcpy async [%asyncToken_17] %memref_16, %arg6 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_19 = arith.constant 0 : index
    %dim_20 = memref.dim %arg7, %c0_19 : memref<?xf64>
    %c2_i32_21 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_21) : (i32) -> ()
    %memref_22, %asyncToken_23 = gpu.alloc async [%4] (%dim_20) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_24 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_24) : (i32) -> ()
    %5 = gpu.memcpy async [%asyncToken_23] %memref_22, %arg7 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg8, %c0_25 : memref<?xf64>
    %c2_i32_27 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_27) : (i32) -> ()
    %memref_28, %asyncToken_29 = gpu.alloc async [%5] (%dim_26) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_30 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_30) : (i32) -> ()
    %6 = gpu.memcpy async [%asyncToken_29] %memref_28, %arg8 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_31 = arith.constant 0 : index
    %dim_32 = memref.dim %arg9, %c0_31 : memref<?xf64>
    %c2_i32_33 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_33) : (i32) -> ()
    %memref_34, %asyncToken_35 = gpu.alloc async [%6] (%dim_32) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_36 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_36) : (i32) -> ()
    %7 = gpu.memcpy async [%asyncToken_35] %memref_34, %arg9 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_37 = arith.constant 0 : index
    %dim_38 = memref.dim %arg10, %c0_37 : memref<?xf64>
    %c2_i32_39 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_39) : (i32) -> ()
    %memref_40, %asyncToken_41 = gpu.alloc async [%7] (%dim_38) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_42 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_42) : (i32) -> ()
    %8 = gpu.memcpy async [%asyncToken_41] %memref_40, %arg10 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c0_43 = arith.constant 0 : index
    %dim_44 = memref.dim %arg11, %c0_43 : memref<?xf64>
    %c2_i32_45 = arith.constant 2 : i32
    call @polygeist_gpu_region_timing_enter(%c2_i32_45) : (i32) -> ()
    %memref_46, %asyncToken_47 = gpu.alloc async [%8] (%dim_44) : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c4_i32_48 = arith.constant 4 : i32
    call @polygeist_gpu_region_timing_enter(%c4_i32_48) : (i32) -> ()
    %9 = gpu.memcpy async [%asyncToken_47] %memref_46, %arg11 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%9]
    %cast = memref.cast %memref : memref<?x?xf64> to memref<*xf64>
    %cast_49 = memref.cast %memref_4 : memref<?xf64> to memref<*xf64>
    %cast_50 = memref.cast %memref_10 : memref<?xf64> to memref<*xf64>
    %cast_51 = memref.cast %memref_16 : memref<?xf64> to memref<*xf64>
    %cast_52 = memref.cast %memref_22 : memref<?xf64> to memref<*xf64>
    %cast_53 = memref.cast %memref_28 : memref<?xf64> to memref<*xf64>
    %cast_54 = memref.cast %memref_34 : memref<?xf64> to memref<*xf64>
    %cast_55 = memref.cast %memref_40 : memref<?xf64> to memref<*xf64>
    %cast_56 = memref.cast %memref_46 : memref<?xf64> to memref<*xf64>
    %c0_57 = arith.constant 0 : index
    %c1_58 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %10 = arith.index_cast %arg0 : i32 to index
    %subview = memref.subview %memref[0, 0] [%10, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_59 = memref.subview %memref_4[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_60 = memref.subview %memref_10[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_61 = memref.subview %memref_16[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_62 = memref.subview %memref_22[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %offset : index to i64
    %13 = arith.muli %12, %c8_i64 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_63 = memref.extract_aligned_pointer_as_index %subview_59 : memref<?xf64, strided<[1]>> -> index
    %16 = arith.index_cast %intptr_63 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %intptr_64 = memref.extract_aligned_pointer_as_index %subview_60 : memref<?xf64, strided<[1]>> -> index
    %18 = arith.index_cast %intptr_64 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %intptr_65 = memref.extract_aligned_pointer_as_index %subview_61 : memref<?xf64, strided<[1]>> -> index
    %20 = arith.index_cast %intptr_65 : index to i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %intptr_66 = memref.extract_aligned_pointer_as_index %subview_62 : memref<?xf64, strided<[1]>> -> index
    %22 = arith.index_cast %intptr_66 : index to i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    call @polygeist_gpu_region_timing_enter(%c1_i32) : (i32) -> ()
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dger_rank2(%arg0, %arg0, %17, %19, %21, %23, %15, %arg0) : (i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %subview_67 = memref.subview %memref[0, 0] [%10, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c1_68 = arith.constant 1 : index
    %24 = affine.apply #map1(%10)[%c0_57, %c1_58]
    %25 = affine.apply #map1(%10)[%c0_57, %c1_58]
    %c0_i64 = arith.constant 0 : i64
    %26 = call @polygeist_cuda_graph_begin(%c0_i64) : (i64) -> i32
    %c0_i32 = arith.constant 0 : i32
    %27 = arith.cmpi ne, %26, %c0_i32 : i32
    scf.if %27 {
      gpu.launch_func  @__polygeist_gpu_module::@kernel_gemver_kernel94051337448048 blocks in (%24, %25, %c1_68) threads in (%c1_68, %c1_68, %c1_68)  args(%c1_58 : index, %c0_57 : index, %subview : memref<?x?xf64, strided<[?, 1], offset: ?>>, %subview_67 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {polygeist.cuda_graph_safe}
      func.call @polygeist_cuda_graph_end(%c0_i64) : (i64) -> ()
    } {polygeist.cuda_graph_scope}
    %subview_69 = memref.subview %memref_40[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_70 = memref.subview %memref_34[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_71 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %28 = arith.index_cast %intptr_71 : index to i64
    %base_buffer_72, %offset_73, %sizes_74:2, %strides_75:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %29 = arith.index_cast %offset_73 : index to i64
    %30 = arith.muli %29, %c8_i64 : i64
    %31 = arith.addi %28, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %intptr_76 = memref.extract_aligned_pointer_as_index %subview_69 : memref<?xf64, strided<[1]>> -> index
    %33 = arith.index_cast %intptr_76 : index to i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %intptr_77 = memref.extract_aligned_pointer_as_index %subview_70 : memref<?xf64, strided<[1]>> -> index
    %35 = arith.index_cast %intptr_77 : index to i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg0, %arg2, %32, %arg0, %34, %cst, %36) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_78 = memref.subview %memref_34[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c1_79 = arith.constant 1 : index
    %37 = affine.apply #map1(%10)[%c0_57, %c1_58]
    %c1_i64 = arith.constant 1 : i64
    %38 = call @polygeist_cuda_graph_begin(%c1_i64) : (i64) -> i32
    %c0_i32_80 = arith.constant 0 : i32
    %39 = arith.cmpi ne, %38, %c0_i32_80 : i32
    scf.if %39 {
      gpu.launch_func  @__polygeist_gpu_module::@kernel_gemver_kernel94051337811360 blocks in (%37, %c1_79, %c1_79) threads in (%c1_79, %c1_79, %c1_79)  args(%c1_58 : index, %c0_57 : index, %subview_70 : memref<?xf64, strided<[1]>>, %subview_78 : memref<?xf64, strided<[1]>>) {polygeist.cuda_graph_safe}
      func.call @polygeist_cuda_graph_end(%c1_i64) : (i64) -> ()
    } {polygeist.cuda_graph_scope}
    %dim_81 = memref.dim %memref_34, %c0_57 : memref<?xf64>
    %40 = arith.index_cast %dim_81 : index to i32
    %intptr_82 = memref.extract_aligned_pointer_as_index %memref_46 : memref<?xf64> -> index
    %41 = arith.index_cast %intptr_82 : index to i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    %intptr_83 = memref.extract_aligned_pointer_as_index %memref_34 : memref<?xf64> -> index
    %43 = arith.index_cast %intptr_83 : index to i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
    call @polygeist_cublas_daxpby(%40, %cst, %42, %cst, %44) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_84 = memref.subview %memref_34[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_85 = memref.subview %memref_28[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_86 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %45 = arith.index_cast %intptr_86 : index to i64
    %base_buffer_87, %offset_88, %sizes_89:2, %strides_90:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %46 = arith.index_cast %offset_88 : index to i64
    %47 = arith.muli %46, %c8_i64 : i64
    %48 = arith.addi %45, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %intptr_91 = memref.extract_aligned_pointer_as_index %subview_84 : memref<?xf64, strided<[1]>> -> index
    %50 = arith.index_cast %intptr_91 : index to i64
    %51 = llvm.inttoptr %50 : i64 to !llvm.ptr
    %intptr_92 = memref.extract_aligned_pointer_as_index %subview_85 : memref<?xf64, strided<[1]>> -> index
    %52 = arith.index_cast %intptr_92 : index to i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg0, %arg1, %49, %arg0, %51, %cst, %53) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %subview_93 = memref.subview %memref_28[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c1_94 = arith.constant 1 : index
    %54 = affine.apply #map1(%10)[%c0_57, %c1_58]
    %c2_i64 = arith.constant 2 : i64
    %55 = call @polygeist_cuda_graph_begin(%c2_i64) : (i64) -> i32
    %c0_i32_95 = arith.constant 0 : i32
    %56 = arith.cmpi ne, %55, %c0_i32_95 : i32
    scf.if %56 {
      gpu.launch_func  @__polygeist_gpu_module::@kernel_gemver_kernel94051337834192 blocks in (%54, %c1_94, %c1_94) threads in (%c1_94, %c1_94, %c1_94)  args(%c1_58 : index, %c0_57 : index, %subview_85 : memref<?xf64, strided<[1]>>, %subview_93 : memref<?xf64, strided<[1]>>) {polygeist.cuda_graph_safe}
      func.call @polygeist_cuda_graph_end(%c2_i64) : (i64) -> ()
    } {polygeist.cuda_graph_scope}
    call @polygeist_cublas_pipeline_end() : () -> ()
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %57 = gpu.wait async
    %c5_i32 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32) : (i32) -> ()
    %58 = gpu.memcpy async [%57] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32) : (i32) -> ()
    %59 = gpu.dealloc async [%58] %memref : memref<?x?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_96 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_96) : (i32) -> ()
    %60 = gpu.memcpy async [%59] %arg4, %memref_4 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_97 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_97) : (i32) -> ()
    %61 = gpu.dealloc async [%60] %memref_4 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_98 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_98) : (i32) -> ()
    %62 = gpu.memcpy async [%61] %arg5, %memref_10 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_99 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_99) : (i32) -> ()
    %63 = gpu.dealloc async [%62] %memref_10 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_100 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_100) : (i32) -> ()
    %64 = gpu.memcpy async [%63] %arg6, %memref_16 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_101 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_101) : (i32) -> ()
    %65 = gpu.dealloc async [%64] %memref_16 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_102 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_102) : (i32) -> ()
    %66 = gpu.memcpy async [%65] %arg7, %memref_22 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_103 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_103) : (i32) -> ()
    %67 = gpu.dealloc async [%66] %memref_22 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_104 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_104) : (i32) -> ()
    %68 = gpu.memcpy async [%67] %arg8, %memref_28 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_105 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_105) : (i32) -> ()
    %69 = gpu.dealloc async [%68] %memref_28 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_106 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_106) : (i32) -> ()
    %70 = gpu.memcpy async [%69] %arg9, %memref_34 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_107 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_107) : (i32) -> ()
    %71 = gpu.dealloc async [%70] %memref_34 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_108 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_108) : (i32) -> ()
    %72 = gpu.memcpy async [%71] %arg10, %memref_40 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_109 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_109) : (i32) -> ()
    %73 = gpu.dealloc async [%72] %memref_40 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c5_i32_110 = arith.constant 5 : i32
    call @polygeist_gpu_region_timing_enter(%c5_i32_110) : (i32) -> ()
    %74 = gpu.memcpy async [%73] %arg11, %memref_46 : memref<?xf64>, memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    %c3_i32_111 = arith.constant 3 : i32
    call @polygeist_gpu_region_timing_enter(%c3_i32_111) : (i32) -> ()
    %75 = gpu.dealloc async [%74] %memref_46 : memref<?xf64>
    call @polygeist_gpu_region_timing_leave() : () -> ()
    gpu.wait [%75]
    call @polygeist_gpu_region_timing_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dger_rank2(i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
  func.func private @polygeist_gpu_region_timing_begin(i64)
  func.func private @polygeist_gpu_region_timing_enter(i32)
  func.func private @polygeist_gpu_region_timing_leave()
  func.func private @polygeist_gpu_region_timing_end()
  func.func private @polygeist_cuda_graph_begin(i64) -> i32
  func.func private @polygeist_cuda_graph_end(i64)
}

