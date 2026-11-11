// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/arith.constant 0 : i32/arith.constant 2 : i32/g' %s > %t.seeded.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seeded.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
module {
  func.func @count_bins(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg0 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%1 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %3 = affine.for %arg2 = 0 to 47 iter_args(%arg3 = %2) -> (tensor<?xi32>) {
      %extracted = tensor.extract %0[%arg2] : tensor<?xi32>
      %5 = arith.index_cast %extracted : i32 to index
      %extracted_0 = tensor.extract %arg3[%5] : tensor<?xi32>
      %6 = arith.addi %extracted_0, %c1_i32 : i32
      %inserted = tensor.insert %6 into %arg3[%5] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

// CHECK: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// CHECK-NOT: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// CHECK-NEGATIVE: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cubHistogramEvenI32ShiftZero_memref
