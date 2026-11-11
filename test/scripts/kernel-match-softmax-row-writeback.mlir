// RUN: sed '/inserted_slice_7 =/s/\[%arg2, 0\]/[%arg2, 1]/' %s > %t.shifted.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.shifted.mlir | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @anonymous_operation(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %2 = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %0) -> (tensor<?x64xf32>) {
      %extracted = tensor.extract %1[%arg2, %c0] : tensor<?x64xf32>
      %alloca = memref.alloca() : memref<f32>
      %4 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %extracted into %4[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %1[%arg2, 1] [1, 63] [1, 1] : tensor<?x64xf32> to tensor<63xf32>
      %5 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<63xf32>) outs(%inserted : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.cmpf ogt, %in, %out : f32
        %10 = arith.select %9, %in, %out : f32
        linalg.yield %10 : f32
      } -> tensor<f32>
      %extracted_0 = tensor.extract %5[] : tensor<f32>
      %alloca_1 = memref.alloca() : memref<f32>
      %6 = bufferization.to_tensor %alloca_1 : memref<f32>
      %inserted_2 = tensor.insert %cst into %6[] : tensor<f32>
      %extracted_slice_3 = tensor.extract_slice %1[%arg2, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_4 = tensor.extract_slice %arg3[%arg2, 0] [1, 64] [1, 1] : tensor<?x64xf32> to tensor<64xf32>
      %7:2 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_3 : tensor<?xf32>) outs(%extracted_slice_4, %inserted_2 : tensor<64xf32>, tensor<f32>) {
      ^bb0(%in: f32, %out: f32, %out_8: f32):
        %9 = arith.subf %in, %extracted_0 : f32
        %10 = math.exp %9 : f32
        %11 = arith.addf %out_8, %10 : f32
        linalg.yield %10, %11 : f32, f32
      } -> (tensor<64xf32>, tensor<f32>)
      %inserted_slice = tensor.insert_slice %7#0 into %arg3[%arg2, 0] [1, 64] [1, 1] : tensor<64xf32> into tensor<?x64xf32>
      %extracted_5 = tensor.extract %7#1[] : tensor<f32>
      %extracted_slice_6 = tensor.extract_slice %inserted_slice[%arg2, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %8 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice_6 : tensor<?xf32>) {
      ^bb0(%out: f32):
        %9 = arith.divf %out, %extracted_5 : f32
        linalg.yield %9 : f32
      } -> tensor<?xf32>
      %inserted_slice_7 = tensor.insert_slice %8 into %inserted_slice[%arg2, 0] [1, %c64] [1, 1] : tensor<?xf32> into tensor<?x64xf32>
      affine.yield %inserted_slice_7 : tensor<?x64xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?x64xf32>
    memref.copy %3, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}


// CHECK: kernel.launch @cudnnSoftmaxForwardOut_tensor
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnSoftmaxForwardOut_tensor
