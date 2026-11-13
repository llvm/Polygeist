// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/constant 1 : i32/constant 2 : i32/g' %s > %t.step.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.step.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/cmpi slt/cmpi ult/g' %s > %t.predicate.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.predicate.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/prefix_stops/other_function/g' %s > %t.rename.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.rename.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  func.func @prefix_stops(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c512_i32 = arith.constant 17 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 12 {
      %0 = affine.load %alloca[] : memref<i32>
      %1 = arith.index_cast %arg2 : index to i32
      %2 = scf.while (%arg3 = %0) : (i32) -> i32 {
        %3 = arith.cmpi slt, %arg3, %c512_i32 : i32
        %4:2 = scf.if %3 -> (i1, i32) {
          %5 = arith.index_cast %arg3 : i32 to index
          %6 = memref.load %arg0[%5] : memref<?xi32>
          %7 = arith.cmpi slt, %6, %1 : i32
          %8 = arith.addi %arg3, %c1_i32 : i32
          %9 = arith.select %7, %8, %arg3 : i32
          scf.yield %7, %9 : i1, i32
        } else {
          scf.yield %false, %arg3 : i1, i32
        }
        scf.condition(%4#0) %4#1 : i32
      } do {
      ^bb0(%arg3: i32):
        scf.yield %arg3 : i32
      }
      affine.store %2, %arg1[%arg2] : memref<?xi32>
      affine.store %2, %alloca[] : memref<i32>
    }
    return
  }
}


// CHECK: kernel.launch @cubPrefixMaxLowerBounds_i32_memref
// CHECK-NOT: scf.while
// CHECK-NEGATIVE-NOT: kernel.launch @cubPrefixMaxLowerBounds_i32_memref
// CHECK-NEGATIVE: scf.while
// CHECK-NEGATIVE-NOT: kernel.launch @cubPrefixMaxLowerBounds_i32_memref
