// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.seed.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seed.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/cmpf ogt/cmpf ugt/g' %s > %t.nan.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.nan.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @joint_absmax(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %cst into %0[] : tensor<f32>
    %1 = tensor.empty() : tensor<f32>
    %inserted_0 = tensor.insert %cst into %1[] : tensor<f32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%c7], strides: [1] : memref<?xf32> to memref<?xf32>
    %2 = bufferization.to_tensor %reinterpret_cast : memref<?xf32>
    %3 = bufferization.to_tensor %arg2 : memref<?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%2 : tensor<?xf32>) outs(%inserted : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf olt, %in, %cst : f32
      %10 = arith.negf %in : f32
      %11 = arith.select %9, %10, %in : f32
      %12 = arith.cmpf ogt, %11, %out : f32
      %13 = arith.select %12, %11, %out : f32
      linalg.yield %13 : f32
    } -> tensor<f32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c7], strides: [1] : memref<?xf32> to memref<?xf32>
    %5 = bufferization.to_tensor %reinterpret_cast_1 : memref<?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%5 : tensor<?xf32>) outs(%inserted_0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf olt, %in, %cst : f32
      %10 = arith.negf %in : f32
      %11 = arith.select %9, %10, %in : f32
      %12 = arith.cmpf ogt, %11, %out : f32
      %13 = arith.select %12, %11, %out : f32
      linalg.yield %13 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %4[] : tensor<f32>
    %extracted_2 = tensor.extract %6[] : tensor<f32>
    %7 = arith.mulf %extracted_2, %extracted : f32
    %inserted_3 = tensor.insert %7 into %3[%c0] : tensor<?xf32>
    %8 = bufferization.to_memref %inserted_3 : memref<?xf32>
    memref.copy %8, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

// CHECK: kernel.launch @cublasJointMaxAbsProduct_f32_memref
// CHECK-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cublasJointMaxAbsProduct_f32_memref
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cublasJointMaxAbsProduct_f32_memref
