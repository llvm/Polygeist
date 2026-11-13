// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/%arg3 + 1/%arg3 + 2/g' %s > %t.invalid.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.invalid.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

module {
  func.func @expand_rows(%offsets: memref<?xi32>, %arg1: memref<?xi32>, %row_indices: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg3 = 0 to 5 {
      %0 = arith.index_cast %arg3 : index to i32
      %1 = affine.load %offsets[%arg3] : memref<?xi32>
      %2 = scf.while (%arg4 = %1) : (i32) -> i32 {
        %3 = affine.load %offsets[%arg3 + 1] : memref<?xi32>
        %4 = arith.cmpi slt, %arg4, %3 : i32
        scf.condition(%4) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):
        %3 = arith.index_cast %arg4 : i32 to index
        memref.store %0, %row_indices[%3] : memref<?xi32>
        %4 = arith.addi %arg4, %c1_i32 : i32
        scf.yield %4 : i32
      }
    }
    return
  }
}

// CHECK: kernel.launch @cusparseXcsr2coo_i32_memref
// CHECK-NOT: scf.while
// CHECK-NEGATIVE-NOT: kernel.launch @cusparseXcsr2coo_i32_memref
// CHECK-NEGATIVE: scf.while
// CHECK-NEGATIVE-NOT: kernel.launch @cusparseXcsr2coo_i32_memref
