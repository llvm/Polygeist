#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg5, %c0 : memref<?xf64>
    %5 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %6 = arith.index_cast %intptr : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg5 : memref<?xf64> -> memref<f64>, index, index, index
    %7 = arith.index_cast %offset : index to i64
    %c8_i64 = arith.constant 8 : i64
    %8 = arith.muli %7, %c8_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_memset_zero_1d(%5, %10) : (i32, !llvm.ptr) -> ()
    %11 = bufferization.to_tensor %arg5 restrict writable : memref<?xf64>
    call @polygeist_cublas_pipeline_end() : () -> ()
    %12 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%0 : tensor<?x?xf64>) outs(%11 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %20 = arith.addf %out, %in : f64
      linalg.yield %20 : f64
    } -> tensor<?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%12 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %20 = arith.divf %out, %arg2 : f64
      linalg.yield %20 : f64
    } -> tensor<?xf64>
    %14 = bufferization.to_memref %13 : memref<?xf64>
    memref.copy %14, %arg5 : memref<?xf64> to memref<?xf64>
    %extracted_slice = tensor.extract_slice %13[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0] [%3, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice : tensor<?xf64>) outs(%extracted_slice_1 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %20 = arith.subf %out, %in : f64
      linalg.yield %20 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %15 into %0[0, 0] [%3, %4] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %16 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %16, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    %17 = arith.subf %arg2, %cst : f64
    %18 = affine.for %arg6 = 0 to %4 iter_args(%arg7 = %1) -> (tensor<?x?xf64>) {
      %20 = affine.for %arg8 = #map2(%arg6) to %4 iter_args(%arg9 = %arg7) -> (tensor<?x?xf64>) {
        %inserted = tensor.insert %cst_0 into %arg9[%arg6, %arg8] : tensor<?x?xf64>
        %extracted_slice_2 = tensor.extract_slice %inserted_slice[0, %arg6] [%3, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_3 = tensor.extract_slice %inserted_slice[0, %arg8] [%3, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_4 = tensor.extract_slice %inserted[%arg6, %arg8] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
        %21 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map5], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_4 : tensor<f64>) {
        ^bb0(%in: f64, %in_8: f64, %out: f64):
          %23 = arith.mulf %in, %in_8 : f64
          %24 = arith.addf %out, %23 : f64
          linalg.yield %24 : f64
        } -> tensor<f64>
        %inserted_slice_5 = tensor.insert_slice %21 into %inserted[%arg6, %arg8] [1, 1] [1, 1] : tensor<f64> into tensor<?x?xf64>
        %extracted = tensor.extract %inserted_slice_5[%arg6, %arg8] : tensor<?x?xf64>
        %22 = arith.divf %extracted, %17 : f64
        %inserted_6 = tensor.insert %22 into %inserted_slice_5[%arg6, %arg8] : tensor<?x?xf64>
        %inserted_7 = tensor.insert %22 into %inserted_6[%arg8, %arg6] : tensor<?x?xf64>
        affine.yield %inserted_7 : tensor<?x?xf64>
      }
      affine.yield %20 : tensor<?x?xf64>
    }
    %19 = bufferization.to_memref %18 : memref<?x?xf64>
    memref.copy %19, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

