#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant -2.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.negf %arg2 : f64
    %4 = math.exp %3 : f64
    %5 = arith.subf %cst_2, %4 : f64
    %6 = arith.mulf %5, %5 : f64
    %7 = arith.mulf %arg2, %cst_1 : f64
    %8 = arith.mulf %7, %4 : f64
    %9 = arith.addf %8, %cst_2 : f64
    %10 = math.exp %7 : f64
    %11 = arith.subf %9, %10 : f64
    %12 = arith.divf %6, %11 : f64
    %13 = arith.mulf %12, %4 : f64
    %14 = arith.subf %arg2, %cst_2 : f64
    %15 = arith.mulf %13, %14 : f64
    %16 = math.powf %cst_1, %3 : f64
    %17 = arith.mulf %arg2, %cst_0 : f64
    %18 = math.exp %17 : f64
    %19 = arith.negf %18 : f64
    %20 = arith.index_cast %arg0 : i32 to index
    %21 = tensor.empty(%20) : tensor<?xf64>
    %22 = tensor.empty(%20) : tensor<?xf64>
    %23 = tensor.empty(%20) : tensor<?xf64>
    %24 = bufferization.to_memref %21 : memref<?xf64>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %24, %c0 : memref<?xf64>
    %25 = arith.index_cast %dim : index to i32
    %26 = bufferization.to_memref %21 : memref<?xf64>
    %intptr = memref.extract_aligned_pointer_as_index %26 : memref<?xf64> -> index
    %27 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %26 : memref<?xf64> -> memref<f64>, index, index, index
    %28 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %29 = arith.muli %28, %c8_i64 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%25, %31) : (i32, !llvm.ptr) -> ()
    %32 = bufferization.to_memref %21 : memref<?xf64>
    %33 = bufferization.to_tensor %32 restrict writable : memref<?xf64>
    %34 = bufferization.to_memref %22 : memref<?xf64>
    %c0_3 = arith.constant 0 : index
    %dim_4 = memref.dim %34, %c0_3 : memref<?xf64>
    %35 = arith.index_cast %dim_4 : index to i32
    %36 = bufferization.to_memref %22 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_5 = memref.extract_aligned_pointer_as_index %36 : memref<?xf64> -> index
    %37 = arith.index_cast %intptr_5 : index to i64
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %36 : memref<?xf64> -> memref<f64>, index, index, index
    %38 = arith.index_cast %offset_7 : index to i64
    %c8_i64_10 = arith.constant 8 : i64
    %39 = arith.muli %38, %c8_i64_10 : i64
    %40 = arith.addi %37, %39 : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%35, %41) : (i32, !llvm.ptr) -> ()
    %42 = bufferization.to_memref %22 : memref<?xf64>
    %43 = bufferization.to_tensor %42 restrict writable : memref<?xf64>
    %44 = bufferization.to_memref %23 : memref<?xf64>
    %c0_11 = arith.constant 0 : index
    %dim_12 = memref.dim %44, %c0_11 : memref<?xf64>
    %45 = arith.index_cast %dim_12 : index to i32
    %46 = bufferization.to_memref %23 : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %intptr_13 = memref.extract_aligned_pointer_as_index %46 : memref<?xf64> -> index
    %47 = arith.index_cast %intptr_13 : index to i64
    %base_buffer_14, %offset_15, %sizes_16, %strides_17 = memref.extract_strided_metadata %46 : memref<?xf64> -> memref<f64>, index, index, index
    %48 = arith.index_cast %offset_15 : index to i64
    %c8_i64_18 = arith.constant 8 : i64
    %49 = arith.muli %48, %c8_i64_18 : i64
    %50 = arith.addi %47, %49 : i64
    %51 = llvm.inttoptr %50 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%45, %51) : (i32, !llvm.ptr) -> ()
    %52 = bufferization.to_memref %23 : memref<?xf64>
    %53 = bufferization.to_tensor %52 restrict writable : memref<?xf64>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_19 = tensor.extract_slice %0[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_20 = tensor.extract_slice %1[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_21 = tensor.extract_slice %53[0] [%20] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_22 = tensor.extract_slice %43[0] [%20] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_23 = tensor.extract_slice %33[0] [%20] [1] : tensor<?xf64> to tensor<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %54:4 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map1, #map1, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_19 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_20, %extracted_slice_21, %extracted_slice_22, %extracted_slice_23 : tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
    ^bb0(%in: f64, %in_24: f64, %out: f64, %out_25: f64, %out_26: f64, %out_27: f64):
      %56 = arith.mulf %12, %in : f64
      %57 = arith.mulf %15, %out_25 : f64
      %58 = arith.addf %56, %57 : f64
      %59 = arith.mulf %16, %out_27 : f64
      %60 = arith.addf %58, %59 : f64
      %61 = arith.mulf %19, %out_26 : f64
      %62 = arith.addf %60, %61 : f64
      linalg.yield %62, %in_24, %out_27, %62 : f64, f64, f64, f64
    } -> (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
    %inserted_slice = tensor.insert_slice %54#0 into %1[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %55 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %55, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

