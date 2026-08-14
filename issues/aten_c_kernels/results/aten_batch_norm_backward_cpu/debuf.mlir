#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_backward_cpu(%arg0: memref<?x8x32xf32>, %arg1: memref<?x8x32xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x8x32xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.280000e+02 : f32
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?x8x32xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?x8x32xf32>
    %5 = bufferization.to_tensor %arg7 : memref<?xf32>
    %6 = bufferization.to_tensor %arg6 : memref<?xf32>
    %7 = bufferization.to_tensor %arg5 : memref<?x8x32xf32>
    %8 = bufferization.to_tensor %arg3 : memref<?xf32>
    %9 = bufferization.to_tensor %arg2 : memref<?xf32>
    %10 = bufferization.to_tensor %arg1 : memref<?x8x32xf32>
    %11 = bufferization.to_tensor %arg0 : memref<?x8x32xf32>
    %12 = tensor.empty(%c8) : tensor<?xf32>
    %13 = tensor.empty(%c8) : tensor<?xf32>
    %14:5 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %12, %arg10 = %13, %arg11 = %7, %arg12 = %6, %arg13 = %5) -> (tensor<?xf32>, tensor<?xf32>, tensor<?x8x32xf32>, tensor<?xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %9[%arg8] : tensor<?xf32>
      %inserted = tensor.insert %cst into %arg9[%arg8] : tensor<?xf32>
      %inserted_1 = tensor.insert %cst into %arg10[%arg8] : tensor<?xf32>
      %alloca = memref.alloca(%c4) : memref<?xf32>
      %18 = bufferization.to_tensor %alloca : memref<?xf32>
      %alloca_2 = memref.alloca(%c4) : memref<?xf32>
      %19 = bufferization.to_tensor %alloca_2 : memref<?xf32>
      %20:4 = affine.for %arg14 = 0 to 4 iter_args(%arg15 = %inserted, %arg16 = %inserted_1, %arg17 = %18, %arg18 = %19) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
        %extracted_13 = tensor.extract %arg15[%arg8] : tensor<?xf32>
        %extracted_14 = tensor.extract %arg16[%arg8] : tensor<?xf32>
        %inserted_15 = tensor.insert %extracted_13 into %arg17[%arg14] : tensor<?xf32>
        %inserted_16 = tensor.insert %extracted_14 into %arg18[%arg14] : tensor<?xf32>
        %extracted_slice_17 = tensor.extract_slice %inserted_15[%arg14] [1] [1] : tensor<?xf32> to tensor<f32>
        %extracted_slice_18 = tensor.extract_slice %inserted_16[%arg14] [1] [1] : tensor<?xf32> to tensor<f32>
        %extracted_slice_19 = tensor.extract_slice %11[%arg14, %arg8, 0] [1, 1, %c32] [1, 1, 1] : tensor<?x8x32xf32> to tensor<?xf32>
        %extracted_slice_20 = tensor.extract_slice %10[%arg14, %arg8, 0] [1, 1, %c32] [1, 1, 1] : tensor<?x8x32xf32> to tensor<?xf32>
        %23:2 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_19, %extracted_slice_20 : tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_17, %extracted_slice_18 : tensor<f32>, tensor<f32>) {
        ^bb0(%in: f32, %in_27: f32, %out: f32, %out_28: f32):
          %24 = arith.addf %out_28, %in : f32
          %25 = arith.subf %in_27, %extracted : f32
          %26 = arith.mulf %in, %25 : f32
          %27 = arith.addf %out, %26 : f32
          linalg.yield %27, %24 : f32, f32
        } -> (tensor<f32>, tensor<f32>)
        %inserted_slice_21 = tensor.insert_slice %23#1 into %inserted_16[%arg14] [1] [1] : tensor<f32> into tensor<?xf32>
        %inserted_slice_22 = tensor.insert_slice %23#0 into %inserted_15[%arg14] [1] [1] : tensor<f32> into tensor<?xf32>
        %extracted_23 = tensor.extract %inserted_slice_22[%arg14] : tensor<?xf32>
        %extracted_24 = tensor.extract %inserted_slice_21[%arg14] : tensor<?xf32>
        %inserted_25 = tensor.insert %extracted_23 into %arg15[%arg8] : tensor<?xf32>
        %inserted_26 = tensor.insert %extracted_24 into %arg16[%arg8] : tensor<?xf32>
        affine.yield %inserted_25, %inserted_26, %inserted_slice_22, %inserted_slice_21 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
      }
      %extracted_3 = tensor.extract %20#0[%arg8] : tensor<?xf32>
      %extracted_4 = tensor.extract %20#1[%arg8] : tensor<?xf32>
      %inserted_5 = tensor.insert %extracted_4 into %arg13[%arg8] : tensor<?xf32>
      %extracted_6 = tensor.extract %8[%arg8] : tensor<?xf32>
      %21 = arith.mulf %extracted_3, %extracted_6 : f32
      %inserted_7 = tensor.insert %21 into %arg12[%arg8] : tensor<?xf32>
      %extracted_slice = tensor.extract_slice %arg11[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : tensor<?x8x32xf32> to tensor<?x?xf32>
      %extracted_slice_8 = tensor.extract_slice %4[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : tensor<?x8x32xf32> to tensor<?x?xf32>
      %extracted_slice_9 = tensor.extract_slice %3[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : tensor<?x8x32xf32> to tensor<?x?xf32>
      %extracted_slice_10 = tensor.extract_slice %2[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_11 = tensor.extract_slice %1[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_12 = tensor.extract_slice %0[%arg8] [1] [1] : tensor<?xf32> to tensor<f32>
      %22 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map3, #map3, #map2, #map3], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_12, %extracted_slice_11, %extracted_slice_8, %extracted_slice_9, %extracted_slice_10 : tensor<f32>, tensor<f32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>) outs(%extracted_slice : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_13: f32, %in_14: f32, %in_15: f32, %in_16: f32, %out: f32):
        %23 = arith.mulf %in, %in_13 : f32
        %24 = arith.divf %23, %cst_0 : f32
        %25 = arith.mulf %in_14, %cst_0 : f32
        %26 = arith.subf %25, %extracted_4 : f32
        %27 = arith.subf %in_15, %in_16 : f32
        %28 = arith.mulf %27, %in_13 : f32
        %29 = arith.mulf %28, %in_13 : f32
        %30 = arith.mulf %29, %extracted_3 : f32
        %31 = arith.subf %26, %30 : f32
        %32 = arith.mulf %24, %31 : f32
        linalg.yield %32 : f32
      } -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %22 into %arg11[0, %arg8, 0] [%c4, 1, %c32] [1, 1, 1] : tensor<?x?xf32> into tensor<?x8x32xf32>
      affine.yield %20#0, %20#1, %inserted_slice, %inserted_7, %inserted_5 : tensor<?xf32>, tensor<?xf32>, tensor<?x8x32xf32>, tensor<?xf32>, tensor<?xf32>
    }
    %15 = bufferization.to_memref %14#4 : memref<?xf32>
    memref.copy %15, %arg7 : memref<?xf32> to memref<?xf32>
    %16 = bufferization.to_memref %14#3 : memref<?xf32>
    memref.copy %16, %arg6 : memref<?xf32> to memref<?xf32>
    %17 = bufferization.to_memref %14#2 : memref<?x8x32xf32>
    memref.copy %17, %arg5 : memref<?x8x32xf32> to memref<?x8x32xf32>
    return
  }
}

