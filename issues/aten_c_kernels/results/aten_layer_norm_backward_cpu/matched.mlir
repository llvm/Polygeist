#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_layer_norm_backward_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x64xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %5 = bufferization.to_tensor %arg7 : memref<?xf32>
    %6 = bufferization.to_tensor %arg6 : memref<?xf32>
    %7 = bufferization.to_tensor %arg5 : memref<?x64xf32>
    %8 = bufferization.to_tensor %arg4 : memref<?xf32>
    %9 = bufferization.to_tensor %arg3 : memref<?xf32>
    %10 = bufferization.to_tensor %arg2 : memref<?xf32>
    %11 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %12 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %13 = kernel.launch @memset_zero_1D_f32(%6) : (tensor<?xf32>) -> tensor<?xf32>
    %14 = kernel.launch @memset_zero_1D_f32(%5) : (tensor<?xf32>) -> tensor<?xf32>
    %15:3 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %7, %arg10 = %13, %arg11 = %14) -> (tensor<?x64xf32>, tensor<?xf32>, tensor<?xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %19 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %19[] : tensor<f32>
      %alloca_1 = memref.alloca() : memref<f32>
      %20 = bufferization.to_tensor %alloca_1 : memref<f32>
      %inserted_2 = tensor.insert %cst into %20[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %12[%arg8, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_3 = tensor.extract_slice %12[%arg8, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_4 = tensor.extract_slice %11[%arg8, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_5 = tensor.extract_slice %10[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_6 = tensor.extract_slice %9[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_7 = tensor.extract_slice %8[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_8 = tensor.extract_slice %arg10[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_9 = tensor.extract_slice %arg11[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
      %21:4 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map1, #map1, #map, #map, #map, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_7, %extracted_slice_4, %extracted_slice_5, %extracted_slice_6, %extracted_slice_3 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<f32>, tensor<f32>, tensor<?xf32>) outs(%extracted_slice_8, %extracted_slice_9, %inserted, %inserted_2 : tensor<?xf32>, tensor<?xf32>, tensor<f32>, tensor<f32>) {
      ^bb0(%in: f32, %in_19: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32, %out_24: f32, %out_25: f32, %out_26: f32):
        %23 = arith.mulf %in, %in_19 : f32
        %24 = arith.addf %out_26, %23 : f32
        %25 = arith.subf %in_20, %in_21 : f32
        %26 = arith.mulf %23, %25 : f32
        %27 = arith.addf %out_25, %26 : f32
        %28 = arith.mulf %in, %25 : f32
        %29 = arith.mulf %28, %in_22 : f32
        %30 = arith.addf %out, %29 : f32
        %31 = arith.addf %out_24, %in_23 : f32
        linalg.yield %30, %31, %27, %24 : f32, f32, f32, f32
      } -> (tensor<?xf32>, tensor<?xf32>, tensor<f32>, tensor<f32>)
      %inserted_slice = tensor.insert_slice %21#1 into %arg11[0] [%c64] [1] : tensor<?xf32> into tensor<?xf32>
      %inserted_slice_10 = tensor.insert_slice %21#0 into %arg10[0] [%c64] [1] : tensor<?xf32> into tensor<?xf32>
      %extracted = tensor.extract %21#2[] : tensor<f32>
      %extracted_11 = tensor.extract %21#3[] : tensor<f32>
      %extracted_slice_12 = tensor.extract_slice %arg9[%arg8, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_13 = tensor.extract_slice %4[%arg8, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_14 = tensor.extract_slice %3[%arg8, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_15 = tensor.extract_slice %2[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_16 = tensor.extract_slice %1[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_17 = tensor.extract_slice %0[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
      %22 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1, #map, #map1, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_13, %extracted_slice_17, %extracted_slice_16, %extracted_slice_14, %extracted_slice_15 : tensor<?xf32>, tensor<?xf32>, tensor<f32>, tensor<?xf32>, tensor<f32>) outs(%extracted_slice_12 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_19: f32, %in_20: f32, %in_21: f32, %in_22: f32, %out: f32):
        %23 = arith.mulf %in, %in_19 : f32
        %24 = arith.divf %in_20, %cst_0 : f32
        %25 = arith.mulf %23, %cst_0 : f32
        %26 = arith.subf %25, %extracted_11 : f32
        %27 = arith.subf %in_21, %in_22 : f32
        %28 = arith.mulf %27, %in_20 : f32
        %29 = arith.mulf %28, %in_20 : f32
        %30 = arith.mulf %29, %extracted : f32
        %31 = arith.subf %26, %30 : f32
        %32 = arith.mulf %24, %31 : f32
        linalg.yield %32 : f32
      } -> tensor<?xf32>
      %inserted_slice_18 = tensor.insert_slice %22 into %arg9[%arg8, 0] [1, %c64] [1, 1] : tensor<?xf32> into tensor<?x64xf32>
      affine.yield %inserted_slice_18, %inserted_slice_10, %inserted_slice : tensor<?x64xf32>, tensor<?xf32>, tensor<?xf32>
    }
    %16 = bufferization.to_memref %15#2 : memref<?xf32>
    memref.copy %16, %arg7 : memref<?xf32> to memref<?xf32>
    %17 = bufferization.to_memref %15#1 : memref<?xf32>
    memref.copy %17, %arg6 : memref<?xf32> to memref<?xf32>
    %18 = bufferization.to_memref %15#0 : memref<?x64xf32>
    memref.copy %18, %arg5 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

