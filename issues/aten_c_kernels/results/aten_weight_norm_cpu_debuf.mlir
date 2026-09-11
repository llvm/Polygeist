#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_weight_norm_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?xf32>, %arg2: memref<?x32xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?xf32>
    %3 = bufferization.to_tensor %arg2 : memref<?x32xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %5:2 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %3, %arg6 = %2) -> (tensor<?x32xf32>, tensor<?xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %8 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %8[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %4[%arg4, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %9 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%inserted : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.mulf %in, %in : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %9[] : tensor<f32>
      %10 = math.sqrt %extracted : f32
      %inserted_0 = tensor.insert %10 into %arg6[%arg4] : tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg5[%arg4, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %extracted_slice_2 = tensor.extract_slice %inserted_0[%arg4] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_3 = tensor.extract_slice %1[%arg4, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %extracted_slice_4 = tensor.extract_slice %0[%arg4] [1] [1] : tensor<?xf32> to tensor<f32>
      %11 = linalg.generic {doc = "", indexing_maps = [#map1, #map, #map1, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_4, %extracted_slice_3, %extracted_slice_2 : tensor<f32>, tensor<?xf32>, tensor<f32>) outs(%extracted_slice_1 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
        %12 = arith.mulf %in, %in_5 : f32
        %13 = arith.divf %12, %in_6 : f32
        linalg.yield %13 : f32
      } -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %11 into %arg5[%arg4, 0] [1, %c32] [1, 1] : tensor<?xf32> into tensor<?x32xf32>
      affine.yield %inserted_slice, %inserted_0 : tensor<?x32xf32>, tensor<?xf32>
    }
    %6 = bufferization.to_memref %5#1 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    %7 = bufferization.to_memref %5#0 : memref<?x32xf32>
    memref.copy %7, %arg2 : memref<?x32xf32> to memref<?x32xf32>
    return
  }
}

