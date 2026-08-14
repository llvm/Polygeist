#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_layer_norm_cpu_backend(%arg0: memref<?x64xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: memref<?x64xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %3 = bufferization.to_tensor %arg6 : memref<?xf32>
    %4 = bufferization.to_tensor %arg5 : memref<?xf32>
    %5 = bufferization.to_tensor %arg4 : memref<?x64xf32>
    %6 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %7:3 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %5, %arg9 = %4, %arg10 = %3) -> (tensor<?x64xf32>, tensor<?xf32>, tensor<?xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %11 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %11[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %6[%arg7, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %12 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%inserted : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.addf %out, %in : f32
        linalg.yield %21 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %12[] : tensor<f32>
      %13 = arith.divf %extracted, %cst_0 : f32
      %inserted_2 = tensor.insert %13 into %arg9[%arg7] : tensor<?xf32>
      %alloca_3 = memref.alloca() : memref<f32>
      %14 = bufferization.to_tensor %alloca_3 : memref<f32>
      %inserted_4 = tensor.insert %cst into %14[] : tensor<f32>
      %extracted_slice_5 = tensor.extract_slice %6[%arg7, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %15 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_5 : tensor<?xf32>) outs(%inserted_4 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.subf %in, %13 : f32
        %22 = arith.mulf %21, %21 : f32
        %23 = arith.addf %out, %22 : f32
        linalg.yield %23 : f32
      } -> tensor<f32>
      %extracted_6 = tensor.extract %15[] : tensor<f32>
      %16 = arith.divf %extracted_6, %cst_0 : f32
      %17 = arith.addf %16, %arg3 : f32
      %18 = math.sqrt %17 : f32
      %19 = arith.divf %cst_1, %18 : f32
      %inserted_7 = tensor.insert %19 into %arg10[%arg7] : tensor<?xf32>
      %extracted_slice_8 = tensor.extract_slice %arg8[%arg7, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_9 = tensor.extract_slice %inserted_2[%arg7] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_10 = tensor.extract_slice %inserted_7[%arg7] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_11 = tensor.extract_slice %2[%arg7, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_12 = tensor.extract_slice %1[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_13 = tensor.extract_slice %0[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
      %20 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map1, #map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_11, %extracted_slice_9, %extracted_slice_10, %extracted_slice_12, %extracted_slice_13 : tensor<?xf32>, tensor<f32>, tensor<f32>, tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_8 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_14: f32, %in_15: f32, %in_16: f32, %in_17: f32, %out: f32):
        %21 = arith.subf %in, %in_14 : f32
        %22 = arith.mulf %21, %in_15 : f32
        %23 = arith.mulf %22, %in_16 : f32
        %24 = arith.addf %23, %in_17 : f32
        linalg.yield %24 : f32
      } -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %20 into %arg8[%arg7, 0] [1, %c64] [1, 1] : tensor<?xf32> into tensor<?x64xf32>
      affine.yield %inserted_slice, %inserted_2, %inserted_7 : tensor<?x64xf32>, tensor<?xf32>, tensor<?xf32>
    }
    %8 = bufferization.to_memref %7#2 : memref<?xf32>
    memref.copy %8, %arg6 : memref<?xf32> to memref<?xf32>
    %9 = bufferization.to_memref %7#1 : memref<?xf32>
    memref.copy %9, %arg5 : memref<?xf32> to memref<?xf32>
    %10 = bufferization.to_memref %7#0 : memref<?x64xf32>
    memref.copy %10, %arg4 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

