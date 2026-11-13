// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/arith.extsi/arith.extui/g' %s > %t.unsigned.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.unsigned.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/constant 0 : i32/constant 1 : i32/g' %s > %t.seed.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seed.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/constant 5 : i32/constant 6 : i32/g' %s > %t.offset.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.offset.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @signed_columns(%arg0: memref<?x7xi8>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c7 = arith.constant 7 : index
    %c5 = arith.constant 5 : index
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg1, %c5_i32 : i32
    %alloca = memref.alloca(%c7) : memref<?xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %subview = memref.subview %arg0[0, 0] [%c5, %c7] [1, 1] : memref<?x7xi8> to memref<?x?xi8, strided<[7, 1]>>
    %reinterpret_cast = memref.reinterpret_cast %alloca to offset: [0], sizes: [%c7], strides: [1] : memref<?xi32> to memref<7xi32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<?x?xi8, strided<[7, 1]>>) outs(%reinterpret_cast : memref<7xi32>) {
    ^bb0(%in: i8, %out: i32):
      %1 = arith.extsi %in : i8 to i32
      %2 = arith.addi %out, %1 : i32
      linalg.yield %2 : i32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca : memref<?xi32>) outs(%arg2 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %1 = arith.subi %in, %0 : i32
      linalg.yield %1 : i32
    }
    return
  }
}

// CHECK: kernel.launch @cubQuantColOffsets_i8_i32_memref
// CHECK-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubQuantColOffsets_i8_i32_memref
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubQuantColOffsets_i8_i32_memref
