// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.invalid.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.invalid.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @row_reduce(%arg0: memref<?x64xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %subview = memref.subview %arg0[0, 0] [%c4, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%c4], strides: [1] : memref<?xf32> to memref<4xf32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<?x?xf32, strided<[64, 1]>>) outs(%reinterpret_cast : memref<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf oeq, %in, %in : f32
      %1 = arith.addf %out, %in : f32
      %2 = arith.select %0, %1, %out : f32
      linalg.yield %2 : f32
    }
    return
  }
}

// CHECK: kernel.launch @cubSegmentedNanSum_f32_memref
// CHECK-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubSegmentedNanSum_f32_memref
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubSegmentedNanSum_f32_memref
