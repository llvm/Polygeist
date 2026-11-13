// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @anonymous_operation(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.store %cst, %arg1[0] : memref<?xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [], strides: [] : memref<?xf32> to memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<?xf32>) outs(%reinterpret_cast : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %out, %in : f32
      linalg.yield %0 : f32
    }
    return
  }
}


// CHECK: kernel.launch @cudnnReduceSum_f32
