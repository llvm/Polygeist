#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flash_attention_cpu(%arg0: memref<?x2x16x32xf32>, %arg1: memref<?x2x16x32xf32>, %arg2: memref<?x2x16x32xf32>, %arg3: memref<?x2x16x32xf32>, %arg4: memref<?x2x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.176776692 : f32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x2x16x32xf32>
    %1 = bufferization.to_tensor %arg4 : memref<?x2x16xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?x2x16x32xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?x2x16x32xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?x2x16x32xf32>
    %5 = tensor.empty() : tensor<16xf32>
    %alloca = memref.alloca() : memref<16xf32>
    %6 = bufferization.to_tensor %alloca : memref<16xf32>
    %7:3 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %5, %arg7 = %2, %arg8 = %1) -> (tensor<16xf32>, tensor<?x2x16x32xf32>, tensor<?x2x16xf32>) {
      %10:3 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %arg6, %arg11 = %arg7, %arg12 = %arg8) -> (tensor<16xf32>, tensor<?x2x16x32xf32>, tensor<?x2x16xf32>) {
        %alloca_2 = memref.alloca() : memref<f32>
        %11 = bufferization.to_tensor %alloca_2 : memref<f32>
        %inserted = tensor.insert %cst into %11[] : tensor<f32>
        %alloca_3 = memref.alloca(%c16) : memref<?xf32>
        %12 = bufferization.to_tensor %alloca_3 : memref<?xf32>
        %13:3 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %arg10, %arg15 = %inserted, %arg16 = %12) -> (tensor<16xf32>, tensor<f32>, tensor<?xf32>) {
          %extracted_12 = tensor.extract %arg15[] : tensor<f32>
          %inserted_13 = tensor.insert %cst_0 into %arg16[%arg13] : tensor<?xf32>
          %extracted_slice_14 = tensor.extract_slice %inserted_13[%arg13] [1] [1] : tensor<?xf32> to tensor<f32>
          %extracted_slice_15 = tensor.extract_slice %4[0, %arg5, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : tensor<?x2x16x32xf32> to tensor<?xf32>
          %extracted_slice_16 = tensor.extract_slice %3[0, %arg5, %arg13, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : tensor<?x2x16x32xf32> to tensor<?xf32>
          %20 = kernel.launch @cublasSdot(%extracted_slice_15, %extracted_slice_16, %extracted_slice_14) : (tensor<?xf32>, tensor<?xf32>, tensor<f32>) -> tensor<f32>
          %inserted_slice_17 = tensor.insert_slice %20 into %inserted_13[%arg13] [1] [1] : tensor<f32> into tensor<?xf32>
          %extracted_18 = tensor.extract %inserted_slice_17[%arg13] : tensor<?xf32>
          %21 = arith.mulf %extracted_18, %cst_1 : f32
          %inserted_19 = tensor.insert %21 into %arg14[%arg13] : tensor<16xf32>
          %22 = arith.cmpf ogt, %21, %extracted_12 : f32
          %23 = arith.select %22, %21, %extracted_12 : f32
          %inserted_20 = tensor.insert %23 into %arg15[] : tensor<f32>
          affine.yield %inserted_19, %inserted_20, %inserted_slice_17 : tensor<16xf32>, tensor<f32>, tensor<?xf32>
        }
        %extracted = tensor.extract %13#1[] : tensor<f32>
        %alloca_4 = memref.alloca() : memref<f32>
        %14 = bufferization.to_tensor %alloca_4 : memref<f32>
        %inserted_5 = tensor.insert %cst_0 into %14[] : tensor<f32>
        %extracted_slice = tensor.extract_slice %13#0[0] [%c16] [1] : tensor<16xf32> to tensor<?xf32>
        %15:2 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} outs(%extracted_slice, %inserted_5 : tensor<?xf32>, tensor<f32>) {
        ^bb0(%out: f32, %out_12: f32):
          %20 = arith.subf %out, %extracted : f32
          %21 = math.exp %20 : f32
          %22 = arith.addf %out_12, %21 : f32
          linalg.yield %21, %22 : f32, f32
        } -> (tensor<?xf32>, tensor<f32>)
        %inserted_slice = tensor.insert_slice %15#0 into %13#0[0] [%c16] [1] : tensor<?xf32> into tensor<16xf32>
        %extracted_6 = tensor.extract %15#1[] : tensor<f32>
        %16 = math.log %extracted_6 : f32
        %17 = arith.addf %extracted, %16 : f32
        %inserted_7 = tensor.insert %17 into %arg12[%c0, %arg5, %arg9] : tensor<?x2x16xf32>
        %extracted_slice_8 = tensor.extract_slice %arg11[0, %arg5, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : tensor<?x2x16x32xf32> to tensor<?xf32>
        %18 = kernel.launch @memset_zero_1D_f32(%extracted_slice_8) : (tensor<?xf32>) -> tensor<?xf32>
        %extracted_slice_9 = tensor.extract_slice %6[0] [%c16] [1] : tensor<16xf32> to tensor<?xf32>
        %extracted_slice_10 = tensor.extract_slice %0[0, %arg5, 0, 0] [1, 1, %c16, %c32] [1, 1, 1, 1] : tensor<?x2x16x32xf32> to tensor<?x?xf32>
        %19 = linalg.generic {doc = "", indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice_9, %extracted_slice_10 : tensor<?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_12: f32, %out: f32):
          %20 = arith.divf %in, %extracted_6 : f32
          %21 = arith.mulf %20, %in_12 : f32
          %22 = arith.addf %out, %21 : f32
          linalg.yield %22 : f32
        } -> tensor<?xf32>
        %inserted_slice_11 = tensor.insert_slice %19 into %arg11[0, %arg5, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : tensor<?xf32> into tensor<?x2x16x32xf32>
        affine.yield %inserted_slice, %inserted_slice_11, %inserted_7 : tensor<16xf32>, tensor<?x2x16x32xf32>, tensor<?x2x16xf32>
      }
      affine.yield %10#0, %10#1, %10#2 : tensor<16xf32>, tensor<?x2x16x32xf32>, tensor<?x2x16xf32>
    }
    %8 = bufferization.to_memref %7#2 : memref<?x2x16xf32>
    memref.copy %8, %arg4 : memref<?x2x16xf32> to memref<?x2x16xf32>
    %9 = bufferization.to_memref %7#1 : memref<?x2x16x32xf32>
    memref.copy %9, %arg3 : memref<?x2x16x32xf32> to memref<?x2x16x32xf32>
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

