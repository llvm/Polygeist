#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gramschmidt(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg2 restrict : memref<?x?xf64>
    %5 = arith.index_cast %arg1 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %7:3 = affine.for %arg5 = 0 to %5 iter_args(%arg6 = %4, %arg7 = %3, %arg8 = %2) -> (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) {
      %alloca = memref.alloca() : memref<f64>
      %11 = bufferization.to_tensor %alloca restrict : memref<f64>
      %inserted = tensor.insert %cst into %11[] : tensor<f64>
      %extracted_slice = tensor.extract_slice %arg6[0, %arg5] [%6, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %12 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf64>) outs(%inserted : tensor<f64>) {
      ^bb0(%in: f64, %out: f64):
        %44 = arith.mulf %in, %in : f64
        %45 = arith.addf %out, %44 : f64
        linalg.yield %45 : f64
      } -> tensor<f64>
      %extracted = tensor.extract %12[] : tensor<f64>
      %13 = math.sqrt %extracted : f64
      %inserted_0 = tensor.insert %13 into %arg7[%arg5, %arg5] : tensor<?x?xf64>
      %extracted_slice_1 = tensor.extract_slice %arg8[0, %arg5] [%6, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_2 = tensor.extract_slice %1[0, %arg5] [%6, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_3 = tensor.extract_slice %0[%arg5, %arg5] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
      %14 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?xf64>, tensor<f64>) outs(%extracted_slice_1 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_23: f64, %out: f64):
        %44 = arith.divf %in, %in_23 : f64
        linalg.yield %44 : f64
      } -> tensor<?xf64>
      %inserted_slice = tensor.insert_slice %14 into %arg8[0, %arg5] [%6, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %extracted_slice_4 = tensor.extract_slice %inserted_0[%arg5, 0] [1, %5] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %15 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice_4 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %44 = linalg.index 0 : index
        %45 = affine.apply #map2(%arg5)
        %46 = arith.cmpi sge, %44, %45 : index
        %47 = arith.select %46, %cst, %out : f64
        linalg.yield %47 : f64
      } -> tensor<?xf64>
      %extracted_slice_5 = tensor.extract_slice %1[0, 0] [%6, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %cst_6 = arith.constant 1.000000e+00 : f64
      %16 = arith.index_cast %6 : index to i32
      %17 = arith.index_cast %5 : index to i32
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<?x?xf64> -> index
      %18 = arith.index_cast %intptr : index to i64
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
      %19 = arith.index_cast %offset : index to i64
      %c0_i64 = arith.constant 0 : i64
      %20 = arith.index_cast %strides#0 : index to i64
      %21 = arith.muli %c0_i64, %20 : i64
      %22 = arith.addi %19, %21 : i64
      %c0_i64_7 = arith.constant 0 : i64
      %23 = arith.index_cast %strides#1 : index to i64
      %24 = arith.muli %c0_i64_7, %23 : i64
      %25 = arith.addi %22, %24 : i64
      %c8_i64 = arith.constant 8 : i64
      %26 = arith.muli %25, %c8_i64 : i64
      %27 = arith.addi %18, %26 : i64
      %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
      %29 = bufferization.to_memref %14 : memref<?xf64>
      %intptr_8 = memref.extract_aligned_pointer_as_index %29 : memref<?xf64> -> index
      %30 = arith.index_cast %intptr_8 : index to i64
      %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %29 : memref<?xf64> -> memref<f64>, index, index, index
      %31 = arith.index_cast %offset_10 : index to i64
      %c8_i64_13 = arith.constant 8 : i64
      %32 = arith.muli %31, %c8_i64_13 : i64
      %33 = arith.addi %30, %32 : i64
      %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
      %35 = bufferization.to_memref %15 : memref<?xf64>
      %intptr_14 = memref.extract_aligned_pointer_as_index %35 : memref<?xf64> -> index
      %36 = arith.index_cast %intptr_14 : index to i64
      %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %35 : memref<?xf64> -> memref<f64>, index, index, index
      %37 = arith.index_cast %offset_16 : index to i64
      %c8_i64_19 = arith.constant 8 : i64
      %38 = arith.muli %37, %c8_i64_19 : i64
      %39 = arith.addi %36, %38 : i64
      %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
      func.call @polygeist_cublas_dgemv_T(%16, %17, %cst_6, %28, %17, %34, %cst_6, %40) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
      %41 = bufferization.to_memref %15 : memref<?xf64>
      %42 = bufferization.to_tensor %41 restrict writable : memref<?xf64>
      %inserted_slice_20 = tensor.insert_slice %42 into %inserted_0[%arg5, 0] [1, %5] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
      %extracted_slice_21 = tensor.extract_slice %arg6[0, 0] [%6, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
      %43 = linalg.generic {doc = "", indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%14, %42 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_21 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_23: f64, %out: f64):
        %44 = arith.mulf %in, %in_23 : f64
        %45 = arith.subf %out, %44 : f64
        linalg.yield %45 : f64
      } -> tensor<?x?xf64>
      %inserted_slice_22 = tensor.insert_slice %43 into %arg6[0, 0] [%6, %5] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
      affine.yield %inserted_slice_22, %inserted_slice_20, %inserted_slice : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>
    }
    %8 = bufferization.to_memref %7#2 : memref<?x?xf64>
    memref.copy %8, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    %9 = bufferization.to_memref %7#1 : memref<?x?xf64>
    memref.copy %9, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %10 = bufferization.to_memref %7#0 : memref<?x?xf64>
    memref.copy %10, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_dgemv_T(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
}

