// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas --canonicalize | FileCheck %s
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  kernel.defn @cublasSsyr2k(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>, %beta: f32, %alpha: f32) -> tensor<?x?xf32> { kernel.yield %c : tensor<?x?xf32> }
  func.func @repeated_slice_bounds(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32, %arg4: memref<?x1200xf32>, %arg5: memref<?x1000xf32>, %arg6: memref<?x1000xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x1200xf32>
    %1 = bufferization.to_tensor %arg5 : memref<?x1000xf32>
    %2 = bufferization.to_tensor %arg6 : memref<?x1000xf32>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.subi %4, %c1 : index
    %6 = affine.apply #map(%5)
    %extracted_slice = tensor.extract_slice %0[0, 0] [%4, %6] [1, 1] : tensor<?x1200xf32> to tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %0[0, 0] [%4, %6] [1, 1] : tensor<?x?xf32> into tensor<?x1200xf32>
    %8 = arith.subi %4, %c1 : index
    %9 = affine.apply #map(%8)
    %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%9, %3] [1, 1] : tensor<?x1000xf32> to tensor<?x?xf32>
    %extracted_slice_1 = tensor.extract_slice %2[0, 0] [%4, %3] [1, 1] : tensor<?x1000xf32> to tensor<?x?xf32>
    %extracted_slice_2 = tensor.extract_slice %2[0, 0] [%9, %3] [1, 1] : tensor<?x1000xf32> to tensor<?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %1[0, 0] [%4, %3] [1, 1] : tensor<?x1000xf32> to tensor<?x?xf32>
    %extracted_slice_4 = tensor.extract_slice %inserted_slice[0, 0] [%4, %9] [1, 1] : tensor<?x1200xf32> to tensor<?x?xf32>
    %10 = kernel.launch @cublasSsyr2k(%extracted_slice_0, %extracted_slice_1, %extracted_slice, %arg3, %arg2) {polygeist.result_destinations = array<i64: 2>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32, f32) -> tensor<?x?xf32>
    %inserted_slice_5 = tensor.insert_slice %10 into %inserted_slice[0, 0] [%4, %9] [1, 1] : tensor<?x?xf32> into tensor<?x1200xf32>
    %11 = bufferization.to_memref %inserted_slice_5 : memref<?x1200xf32>
    memref.copy %11, %arg4 : memref<?x1200xf32> to memref<?x1200xf32>
    return
  }
}
// CHECK-LABEL: func.func @repeated_slice_bounds
// CHECK: call @polygeist_cublas_ssyr2k_lower
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: memref.alloc
// CHECK-NOT: call @polygeist_cuda_copy
// CHECK: return
