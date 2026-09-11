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
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.gpu_data_residency, polygeist.gpu_residual_pipeline} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf64>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] (%dim, %dim_0) : memref<?x?xf64>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg3 : memref<?x?xf64>, memref<?x?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg4, %c0_1 : memref<?xf64>
    %memref_3, %asyncToken_4 = gpu.alloc async [%1] (%dim_2) : memref<?xf64>
    %2 = gpu.memcpy async [%asyncToken_4] %memref_3, %arg4 : memref<?xf64>, memref<?xf64>
    %c0_5 = arith.constant 0 : index
    %dim_6 = memref.dim %arg5, %c0_5 : memref<?xf64>
    %memref_7, %asyncToken_8 = gpu.alloc async [%2] (%dim_6) : memref<?xf64>
    %3 = gpu.memcpy async [%asyncToken_8] %memref_7, %arg5 : memref<?xf64>, memref<?xf64>
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg6, %c0_9 : memref<?xf64>
    %memref_11, %asyncToken_12 = gpu.alloc async [%3] (%dim_10) : memref<?xf64>
    %4 = gpu.memcpy async [%asyncToken_12] %memref_11, %arg6 : memref<?xf64>, memref<?xf64>
    %c0_13 = arith.constant 0 : index
    %dim_14 = memref.dim %arg7, %c0_13 : memref<?xf64>
    %memref_15, %asyncToken_16 = gpu.alloc async [%4] (%dim_14) : memref<?xf64>
    %5 = gpu.memcpy async [%asyncToken_16] %memref_15, %arg7 : memref<?xf64>, memref<?xf64>
    %c0_17 = arith.constant 0 : index
    %dim_18 = memref.dim %arg8, %c0_17 : memref<?xf64>
    %memref_19, %asyncToken_20 = gpu.alloc async [%5] (%dim_18) : memref<?xf64>
    %6 = gpu.memcpy async [%asyncToken_20] %memref_19, %arg8 : memref<?xf64>, memref<?xf64>
    %c0_21 = arith.constant 0 : index
    %dim_22 = memref.dim %arg9, %c0_21 : memref<?xf64>
    %memref_23, %asyncToken_24 = gpu.alloc async [%6] (%dim_22) : memref<?xf64>
    %7 = gpu.memcpy async [%asyncToken_24] %memref_23, %arg9 : memref<?xf64>, memref<?xf64>
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg10, %c0_25 : memref<?xf64>
    %memref_27, %asyncToken_28 = gpu.alloc async [%7] (%dim_26) : memref<?xf64>
    %8 = gpu.memcpy async [%asyncToken_28] %memref_27, %arg10 : memref<?xf64>, memref<?xf64>
    %c0_29 = arith.constant 0 : index
    %dim_30 = memref.dim %arg11, %c0_29 : memref<?xf64>
    %memref_31, %asyncToken_32 = gpu.alloc async [%8] (%dim_30) : memref<?xf64>
    %9 = gpu.memcpy async [%asyncToken_32] %memref_31, %arg11 : memref<?xf64>, memref<?xf64>
    gpu.wait [%9]
    %cast = memref.cast %memref : memref<?x?xf64> to memref<*xf64>
    %cast_33 = memref.cast %memref_3 : memref<?xf64> to memref<*xf64>
    %cast_34 = memref.cast %memref_7 : memref<?xf64> to memref<*xf64>
    %cast_35 = memref.cast %memref_11 : memref<?xf64> to memref<*xf64>
    %cast_36 = memref.cast %memref_15 : memref<?xf64> to memref<*xf64>
    %cast_37 = memref.cast %memref_19 : memref<?xf64> to memref<*xf64>
    %cast_38 = memref.cast %memref_23 : memref<?xf64> to memref<*xf64>
    %cast_39 = memref.cast %memref_27 : memref<?xf64> to memref<*xf64>
    %cast_40 = memref.cast %memref_31 : memref<?xf64> to memref<*xf64>
    %c0_41 = arith.constant 0 : index
    %c1_42 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c8_i64 = arith.constant 8 : i64
    %10 = arith.index_cast %arg0 : i32 to index
    %subview = memref.subview %memref[0, 0] [%10, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %subview_43 = memref.subview %memref_3[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_44 = memref.subview %memref_7[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_45 = memref.subview %memref_11[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_46 = memref.subview %memref_15[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %11 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %12 = arith.index_cast %offset : index to i64
    %13 = arith.muli %12, %c8_i64 : i64
    %14 = arith.addi %11, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %intptr_47 = memref.extract_aligned_pointer_as_index %subview_43 : memref<?xf64, strided<[1]>> -> index
    %16 = arith.index_cast %intptr_47 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %intptr_48 = memref.extract_aligned_pointer_as_index %subview_44 : memref<?xf64, strided<[1]>> -> index
    %18 = arith.index_cast %intptr_48 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %intptr_49 = memref.extract_aligned_pointer_as_index %subview_45 : memref<?xf64, strided<[1]>> -> index
    %20 = arith.index_cast %intptr_49 : index to i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %intptr_50 = memref.extract_aligned_pointer_as_index %subview_46 : memref<?xf64, strided<[1]>> -> index
    %22 = arith.index_cast %intptr_50 : index to i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dger_rank2(%arg0, %arg0, %17, %19, %21, %23, %15, %arg0) : (i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %subview_51 = memref.subview %memref[0, 0] [%10, %10] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %c1_52 = arith.constant 1 : index
    %24 = affine.apply #map1(%10)[%c0_41, %c1_42]
    %25 = affine.apply #map1(%10)[%c0_41, %c1_42]
    gpu.launch_func  @__polygeist_gpu_module::@kernel_gemver_kernel94051337448048 blocks in (%24, %25, %c1_52) threads in (%c1_52, %c1_52, %c1_52)  args(%c1_42 : index, %c0_41 : index, %subview : memref<?x?xf64, strided<[?, 1], offset: ?>>, %subview_51 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {polygeist.cuda_graph_safe}
    %subview_53 = memref.subview %memref_27[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_54 = memref.subview %memref_23[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_55 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %26 = arith.index_cast %intptr_55 : index to i64
    %base_buffer_56, %offset_57, %sizes_58:2, %strides_59:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %27 = arith.index_cast %offset_57 : index to i64
    %28 = arith.muli %27, %c8_i64 : i64
    %29 = arith.addi %26, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %intptr_60 = memref.extract_aligned_pointer_as_index %subview_53 : memref<?xf64, strided<[1]>> -> index
    %31 = arith.index_cast %intptr_60 : index to i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %intptr_61 = memref.extract_aligned_pointer_as_index %subview_54 : memref<?xf64, strided<[1]>> -> index
    %33 = arith.index_cast %intptr_61 : index to i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg0, %arg0, %arg2, %30, %arg0, %32, %cst, %34) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %subview_62 = memref.subview %memref_23[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c1_63 = arith.constant 1 : index
    %35 = affine.apply #map1(%10)[%c0_41, %c1_42]
    gpu.launch_func  @__polygeist_gpu_module::@kernel_gemver_kernel94051337811360 blocks in (%35, %c1_63, %c1_63) threads in (%c1_63, %c1_63, %c1_63)  args(%c1_42 : index, %c0_41 : index, %subview_54 : memref<?xf64, strided<[1]>>, %subview_62 : memref<?xf64, strided<[1]>>) {polygeist.cuda_graph_safe}
    %dim_64 = memref.dim %memref_23, %c0_41 : memref<?xf64>
    %36 = arith.index_cast %dim_64 : index to i32
    %intptr_65 = memref.extract_aligned_pointer_as_index %memref_31 : memref<?xf64> -> index
    %37 = arith.index_cast %intptr_65 : index to i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %intptr_66 = memref.extract_aligned_pointer_as_index %memref_23 : memref<?xf64> -> index
    %39 = arith.index_cast %intptr_66 : index to i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_daxpby(%36, %cst, %38, %cst, %40) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %subview_67 = memref.subview %memref_23[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %subview_68 = memref.subview %memref_19[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_69 = memref.extract_aligned_pointer_as_index %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> index
    %41 = arith.index_cast %intptr_69 : index to i64
    %base_buffer_70, %offset_71, %sizes_72:2, %strides_73:2 = memref.extract_strided_metadata %subview : memref<?x?xf64, strided<[?, 1], offset: ?>> -> memref<f64>, index, index, index, index, index
    %42 = arith.index_cast %offset_71 : index to i64
    %43 = arith.muli %42, %c8_i64 : i64
    %44 = arith.addi %41, %43 : i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_74 = memref.extract_aligned_pointer_as_index %subview_67 : memref<?xf64, strided<[1]>> -> index
    %46 = arith.index_cast %intptr_74 : index to i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    %intptr_75 = memref.extract_aligned_pointer_as_index %subview_68 : memref<?xf64, strided<[1]>> -> index
    %48 = arith.index_cast %intptr_75 : index to i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dgemv(%arg0, %arg0, %arg1, %45, %arg0, %47, %cst, %49) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    %subview_76 = memref.subview %memref_19[0] [%10] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %c1_77 = arith.constant 1 : index
    %50 = affine.apply #map1(%10)[%c0_41, %c1_42]
    gpu.launch_func  @__polygeist_gpu_module::@kernel_gemver_kernel94051337834192 blocks in (%50, %c1_77, %c1_77) threads in (%c1_77, %c1_77, %c1_77)  args(%c1_42 : index, %c0_41 : index, %subview_68 : memref<?xf64, strided<[1]>>, %subview_76 : memref<?xf64, strided<[1]>>) {polygeist.cuda_graph_safe}
    %51 = gpu.wait async
    %52 = gpu.memcpy async [%51] %arg3, %memref : memref<?x?xf64>, memref<?x?xf64>
    %53 = gpu.dealloc async [%52] %memref : memref<?x?xf64>
    %54 = gpu.memcpy async [%53] %arg4, %memref_3 : memref<?xf64>, memref<?xf64>
    %55 = gpu.dealloc async [%54] %memref_3 : memref<?xf64>
    %56 = gpu.memcpy async [%55] %arg5, %memref_7 : memref<?xf64>, memref<?xf64>
    %57 = gpu.dealloc async [%56] %memref_7 : memref<?xf64>
    %58 = gpu.memcpy async [%57] %arg6, %memref_11 : memref<?xf64>, memref<?xf64>
    %59 = gpu.dealloc async [%58] %memref_11 : memref<?xf64>
    %60 = gpu.memcpy async [%59] %arg7, %memref_15 : memref<?xf64>, memref<?xf64>
    %61 = gpu.dealloc async [%60] %memref_15 : memref<?xf64>
    %62 = gpu.memcpy async [%61] %arg8, %memref_19 : memref<?xf64>, memref<?xf64>
    %63 = gpu.dealloc async [%62] %memref_19 : memref<?xf64>
    %64 = gpu.memcpy async [%63] %arg9, %memref_23 : memref<?xf64>, memref<?xf64>
    %65 = gpu.dealloc async [%64] %memref_23 : memref<?xf64>
    %66 = gpu.memcpy async [%65] %arg10, %memref_27 : memref<?xf64>, memref<?xf64>
    %67 = gpu.dealloc async [%66] %memref_27 : memref<?xf64>
    %68 = gpu.memcpy async [%67] %arg11, %memref_31 : memref<?xf64>, memref<?xf64>
    %69 = gpu.dealloc async [%68] %memref_31 : memref<?xf64>
    gpu.wait [%69]
    return
  }
  func.func private @polygeist_cublas_dger_rank2(i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

