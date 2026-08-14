#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_weight_norm_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x32xf32>, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?x32xf32>
    %3 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %4 = bufferization.to_tensor %arg5 : memref<?xf32>
    %5 = bufferization.to_tensor %arg4 : memref<?x32xf32>
    %6 = bufferization.to_tensor %arg3 : memref<?xf32>
    %7 = bufferization.to_tensor %arg1 : memref<?x32xf32>
    %8 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %9:2 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %5, %arg8 = %4) -> (tensor<?x32xf32>, tensor<?xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %12 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %12[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %8[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %7[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %13 = kernel.launch @cublasSdot(%extracted_slice, %extracted_slice_0, %inserted) : (tensor<?xf32>, tensor<?xf32>, tensor<f32>) -> tensor<f32>
      %extracted = tensor.extract %13[] : tensor<f32>
      %extracted_1 = tensor.extract %6[%arg6] : tensor<?xf32>
      %14 = arith.divf %extracted, %extracted_1 : f32
      %inserted_2 = tensor.insert %14 into %arg8[%arg6] : tensor<?xf32>
      %extracted_slice_3 = tensor.extract_slice %arg7[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %extracted_slice_4 = tensor.extract_slice %3[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %extracted_slice_5 = tensor.extract_slice %2[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %extracted_slice_6 = tensor.extract_slice %1[%arg6] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_7 = tensor.extract_slice %0[%arg6] [1] [1] : tensor<?xf32> to tensor<f32>
      %15 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_6, %extracted_slice_7, %extracted_slice_4, %extracted_slice_5 : tensor<f32>, tensor<f32>, tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_3 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %in_9: f32, %in_10: f32, %out: f32):
        %16 = arith.divf %in, %in_8 : f32
        %17 = arith.mulf %in_10, %extracted : f32
        %18 = arith.mulf %in_8, %in_8 : f32
        %19 = arith.divf %17, %18 : f32
        %20 = arith.subf %in_9, %19 : f32
        %21 = arith.mulf %16, %20 : f32
        linalg.yield %21 : f32
      } -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %15 into %arg7[%arg6, 0] [1, %c32] [1, 1] : tensor<?xf32> into tensor<?x32xf32>
      affine.yield %inserted_slice, %inserted_2 : tensor<?x32xf32>, tensor<?xf32>
    }
    %10 = bufferization.to_memref %9#1 : memref<?xf32>
    memref.copy %10, %arg5 : memref<?xf32> to memref<?xf32>
    %11 = bufferization.to_memref %9#0 : memref<?x32xf32>
    memref.copy %11, %arg4 : memref<?x32xf32> to memref<?x32xf32>
    return
  }
}

