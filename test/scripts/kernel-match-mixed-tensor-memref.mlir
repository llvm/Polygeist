// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @anonymous_operation(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.280000e+02 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %cst into %4[] : tensor<f32>
    %inserted_2 = tensor.insert %cst into %3[] : tensor<f32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c128], strides: [1] : memref<?xf32> to memref<?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%reinterpret_cast : memref<?xf32>) outs(%inserted : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %out, %in : f32
      linalg.yield %14 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %5[] : tensor<f32>
    %6 = arith.divf %extracted, %cst_0 : f32
    %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c128], strides: [1] : memref<?xf32> to memref<?xf32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%reinterpret_cast_3 : memref<?xf32>) outs(%inserted_2 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.subf %in, %6 : f32
      %15 = arith.mulf %14, %14 : f32
      %16 = arith.addf %out, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<f32>
    %extracted_4 = tensor.extract %7[] : tensor<f32>
    %8 = arith.divf %extracted_4, %cst_0 : f32
    %9 = arith.addf %8, %arg4 : f32
    %10 = math.sqrt %9 : f32
    %11 = arith.divf %cst_1, %10 : f32
    %12 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%arg0, %1, %0 : memref<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
      %14 = arith.subf %in, %6 : f32
      %15 = arith.mulf %14, %11 : f32
      %16 = arith.mulf %15, %in_5 : f32
      %17 = arith.addf %16, %in_6 : f32
      linalg.yield %17 : f32
    } -> tensor<?xf32>
    %13 = bufferization.to_memref %12 : memref<?xf32>
    memref.copy %13, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}


// CHECK: kernel.launch @cudnnPointwiseGraph_f32
