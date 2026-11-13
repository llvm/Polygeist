// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.seed.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seed.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/arith.addf/arith.subf/g' %s > %t.operation.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.operation.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

// RUN: sed "s/scatter_plane/unrelated_name/g" %s > %t.rename.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.rename.mlir --enable-structured-rewrite | sed "/^\/\/ CHECK/d" | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @scatter_plane(%arg0: memref<?x5x7x7xf32>, %arg1: memref<?x11x3x3xf32>, %arg2: memref<?x11x9x9xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c3 = arith.constant 3 : index
    %c130 = arith.constant 9 : index
    %c32 = arith.constant 11 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x5x7x7xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x11x3x3xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x11x9x9xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0, 0, 0] [%c2, %c32, %c130, %c130] [1, 1, 1, 1] : tensor<?x11x9x9xf32> to tensor<?x?x?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %2[0, 0, 0, 0] [%c2, %c32, %c130, %c130] [1, 1, 1, 1] : tensor<?x?x?x?xf32> into tensor<?x11x9x9xf32>
    %4 = affine.for %arg3 = 0 to 2 iter_args(%arg4 = %inserted_slice) -> (tensor<?x11x9x9xf32>) {
      %6 = affine.for %arg5 = 0 to 5 iter_args(%arg6 = %arg4) -> (tensor<?x11x9x9xf32>) {
        %7 = affine.for %arg7 = 0 to 7 iter_args(%arg8 = %arg6) -> (tensor<?x11x9x9xf32>) {
          %8 = affine.for %arg9 = 0 to 7 iter_args(%arg10 = %arg8) -> (tensor<?x11x9x9xf32>) {
            %extracted_slice_0 = tensor.extract_slice %0[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<?x5x7x7xf32> to tensor<f32>
            %extracted_slice_1 = tensor.extract_slice %1[%arg5, 0, 0, 0] [1, %c32, %c3, %c3] [1, 1, 1, 1] : tensor<?x11x3x3xf32> to tensor<?x?x?xf32>
            %extracted_slice_2 = tensor.extract_slice %arg10[%arg3, 0, %arg7, %arg9] [1, %c32, %c3, %c3] [1, 1, 1, 1] : tensor<?x11x9x9xf32> to tensor<?x?x?xf32>
            %9 = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%extracted_slice_0, %extracted_slice_1 : tensor<f32>, tensor<?x?x?xf32>) outs(%extracted_slice_2 : tensor<?x?x?xf32>) {
            ^bb0(%in: f32, %in_4: f32, %out: f32):
              %10 = arith.mulf %in, %in_4 : f32
              %11 = arith.addf %out, %10 : f32
              linalg.yield %11 : f32
            } -> tensor<?x?x?xf32>
            %inserted_slice_3 = tensor.insert_slice %9 into %arg10[%arg3, 0, %arg7, %arg9] [1, %c32, %c3, %c3] [1, 1, 1, 1] : tensor<?x?x?xf32> into tensor<?x11x9x9xf32>
            affine.yield %inserted_slice_3 : tensor<?x11x9x9xf32>
          }
          affine.yield %8 : tensor<?x11x9x9xf32>
        }
        affine.yield %7 : tensor<?x11x9x9xf32>
      }
      affine.yield %6 : tensor<?x11x9x9xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?x11x9x9xf32>
    memref.copy %5, %arg2 : memref<?x11x9x9xf32> to memref<?x11x9x9xf32>
    return
  }
}

// CHECK: kernel.launch @cudnnConvolutionTranspose2D_f32_memref
// CHECK-NOT: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnConvolutionTranspose2D_f32_memref
// CHECK-NEGATIVE: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnConvolutionTranspose2D_f32_memref
