#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_group_norm_cpu(%arg0: memref<?x4x2x16xf32>, %arg1: memref<?x2xf32>, %arg2: memref<?x2xf32>, %arg3: f32, %arg4: memref<?x4x2x16xf32>, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 3.200000e+01 : f32
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x2xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x2xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x4x2x16xf32>
    %3 = bufferization.to_tensor %arg6 : memref<?x4xf32>
    %4 = bufferization.to_tensor %arg5 : memref<?x4xf32>
    %5 = bufferization.to_tensor %arg4 : memref<?x4x2x16xf32>
    %6 = bufferization.to_tensor %arg0 : memref<?x4x2x16xf32>
    %7:3 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %5, %arg9 = %4, %arg10 = %3) -> (tensor<?x4x2x16xf32>, tensor<?x4xf32>, tensor<?x4xf32>) {
      %11:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %arg8, %arg13 = %arg9, %arg14 = %arg10) -> (tensor<?x4x2x16xf32>, tensor<?x4xf32>, tensor<?x4xf32>) {
        %alloca = memref.alloca() : memref<f32>
        %12 = bufferization.to_tensor %alloca : memref<f32>
        %inserted = tensor.insert %cst into %12[] : tensor<f32>
        %extracted_slice = tensor.extract_slice %6[%arg7, %arg11, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : tensor<?x4x2x16xf32> to tensor<?x?xf32>
        %13 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"], library_call = ""} ins(%extracted_slice : tensor<?x?xf32>) outs(%inserted : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %22 = arith.addf %out, %in : f32
          linalg.yield %22 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %13[] : tensor<f32>
        %14 = arith.divf %extracted, %cst_1 : f32
        %inserted_2 = tensor.insert %14 into %arg13[%arg7, %arg11] : tensor<?x4xf32>
        %alloca_3 = memref.alloca() : memref<f32>
        %15 = bufferization.to_tensor %alloca_3 : memref<f32>
        %inserted_4 = tensor.insert %cst into %15[] : tensor<f32>
        %extracted_slice_5 = tensor.extract_slice %6[%arg7, %arg11, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : tensor<?x4x2x16xf32> to tensor<?x?xf32>
        %16 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"], library_call = ""} ins(%extracted_slice_5 : tensor<?x?xf32>) outs(%inserted_4 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %22 = arith.subf %in, %14 : f32
          %23 = arith.mulf %22, %22 : f32
          %24 = arith.addf %out, %23 : f32
          linalg.yield %24 : f32
        } -> tensor<f32>
        %extracted_6 = tensor.extract %16[] : tensor<f32>
        %17 = arith.divf %extracted_6, %cst_1 : f32
        %18 = arith.addf %17, %arg3 : f32
        %19 = math.sqrt %18 : f32
        %20 = arith.divf %cst_0, %19 : f32
        %inserted_7 = tensor.insert %20 into %arg14[%arg7, %arg11] : tensor<?x4xf32>
        %extracted_slice_8 = tensor.extract_slice %arg12[%arg7, %arg11, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : tensor<?x4x2x16xf32> to tensor<?x?xf32>
        %extracted_slice_9 = tensor.extract_slice %inserted_2[%arg7, %arg11] [1, 1] [1, 1] : tensor<?x4xf32> to tensor<f32>
        %extracted_slice_10 = tensor.extract_slice %inserted_7[%arg7, %arg11] [1, 1] [1, 1] : tensor<?x4xf32> to tensor<f32>
        %extracted_slice_11 = tensor.extract_slice %2[%arg7, %arg11, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : tensor<?x4x2x16xf32> to tensor<?x?xf32>
        %extracted_slice_12 = tensor.extract_slice %1[%arg11, 0] [1, %c2] [1, 1] : tensor<?x2xf32> to tensor<?xf32>
        %extracted_slice_13 = tensor.extract_slice %0[%arg11, 0] [1, %c2] [1, 1] : tensor<?x2xf32> to tensor<?xf32>
        %21 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map1, #map2, #map2, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%extracted_slice_11, %extracted_slice_9, %extracted_slice_10, %extracted_slice_12, %extracted_slice_13 : tensor<?x?xf32>, tensor<f32>, tensor<f32>, tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_8 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_14: f32, %in_15: f32, %in_16: f32, %in_17: f32, %out: f32):
          %22 = arith.subf %in, %in_14 : f32
          %23 = arith.mulf %22, %in_15 : f32
          %24 = arith.mulf %23, %in_16 : f32
          %25 = arith.addf %24, %in_17 : f32
          linalg.yield %25 : f32
        } -> tensor<?x?xf32>
        %inserted_slice = tensor.insert_slice %21 into %arg12[%arg7, %arg11, 0, 0] [1, 1, %c2, %c16] [1, 1, 1, 1] : tensor<?x?xf32> into tensor<?x4x2x16xf32>
        affine.yield %inserted_slice, %inserted_2, %inserted_7 : tensor<?x4x2x16xf32>, tensor<?x4xf32>, tensor<?x4xf32>
      }
      affine.yield %11#0, %11#1, %11#2 : tensor<?x4x2x16xf32>, tensor<?x4xf32>, tensor<?x4xf32>
    }
    %8 = bufferization.to_memref %7#2 : memref<?x4xf32>
    memref.copy %8, %arg6 : memref<?x4xf32> to memref<?x4xf32>
    %9 = bufferization.to_memref %7#1 : memref<?x4xf32>
    memref.copy %9, %arg5 : memref<?x4xf32> to memref<?x4xf32>
    %10 = bufferization.to_memref %7#0 : memref<?x4x2x16xf32>
    memref.copy %10, %arg4 : memref<?x4x2x16xf32> to memref<?x4x2x16xf32>
    return
  }
}

