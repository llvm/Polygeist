// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s
module {
  func.func @copy_column(%x: memref<?xf32, strided<[64], offset: ?>>, %y: memref<?xf32, strided<[1], offset: ?>>) {
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : memref<?xf32, strided<[64], offset: ?>>) outs(%y : memref<?xf32, strided<[1], offset: ?>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}
// CHECK: memref.cast %x : memref<?xf32, strided<[64], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
// CHECK: kernel.launch @cudaCopy1D_f32_memref
// CHECK-NOT: linalg.generic
