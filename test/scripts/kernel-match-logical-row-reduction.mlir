// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/arith.extui/arith.extsi/g' %s > %t.invalid.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.invalid.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @row_reduce(%arg0: memref<?x64xi32>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg1, %c0_i32 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg2 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %arg1 : i32
    }
    %subview = memref.subview %arg0[0, 0] [%c4, %c64] [1, 1] : memref<?x64xi32> to memref<?x?xi32, strided<[64, 1]>>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%c4], strides: [1] : memref<?xi32> to memref<4xi32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<?x?xi32, strided<[64, 1]>>) outs(%reinterpret_cast : memref<4xi32>) {
    ^bb0(%in: i32, %out: i32):
      %1 = arith.cmpi ne, %out, %c0_i32 : i32
      %2 = arith.cmpi ne, %in, %c0_i32 : i32
      %3 = arith.select %1, %2, %false : i1
      %4 = arith.extui %3 : i1 to i32
      %5 = arith.cmpi ne, %out, %c0_i32 : i32
      %6 = arith.cmpi ne, %in, %c0_i32 : i32
      %7 = arith.select %5, %true, %6 : i1
      %8 = arith.extui %7 : i1 to i32
      %9 = arith.select %0, %4, %8 : i32
      linalg.yield %9 : i32
    }
    return
  }
}

// CHECK: kernel.launch @cubSegmentedLogicalSelect_i32_memref
// CHECK-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubSegmentedLogicalSelect_i32_memref
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubSegmentedLogicalSelect_i32_memref
