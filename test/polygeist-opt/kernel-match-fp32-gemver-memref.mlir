// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#id1 = affine_map<(d0) -> (d0)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
#mat = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

module {
  // CHECK-LABEL: func.func @gemver_memref
  // CHECK: kernel.launch @cublasSger_rank2_memref
  // CHECK: kernel.launch @cublasSgemv_alpha_T_memref
  // CHECK: kernel.launch @cublasSaxpby_memref
  // CHECK: kernel.launch @cublasSgemv_alpha_memref
  func.func @gemver_memref(
      %a: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %u1: memref<?xf32, strided<[1], offset: ?>>,
      %v1: memref<?xf32, strided<[1], offset: ?>>,
      %u2: memref<?xf32, strided<[1], offset: ?>>,
      %v2: memref<?xf32, strided<[1], offset: ?>>,
      %x: memref<?xf32>, %y: memref<?xf32>, %z: memref<?xf32>,
      %w: memref<?xf32>, %alpha: f32, %beta: f32) {
    linalg.generic {
      indexing_maps = [#row, #col, #row, #col, #mat],
      iterator_types = ["parallel", "parallel"]
    } ins(%u1, %v1, %u2, %v2
          : memref<?xf32, strided<[1], offset: ?>>,
            memref<?xf32, strided<[1], offset: ?>>,
            memref<?xf32, strided<[1], offset: ?>>,
            memref<?xf32, strided<[1], offset: ?>>)
      outs(%a : memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    ^bb0(%u1v: f32, %v1v: f32, %u2v: f32, %v2v: f32, %out: f32):
      %p1 = arith.mulf %u1v, %v1v : f32
      %s1 = arith.addf %out, %p1 : f32
      %p2 = arith.mulf %u2v, %v2v : f32
      %s2 = arith.addf %s1, %p2 : f32
      linalg.yield %s2 : f32
    }
    linalg.generic {
      indexing_maps = [#transpose, #col, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %y : memref<?x?xf32, strided<[?, 1], offset: ?>>,
                   memref<?xf32>) outs(%x : memref<?xf32>) {
    ^bb0(%av: f32, %yv: f32, %out: f32):
      %scaled = arith.mulf %beta, %av : f32
      %product = arith.mulf %scaled, %yv : f32
      %sum = arith.addf %out, %product : f32
      linalg.yield %sum : f32
    }
    linalg.generic {indexing_maps = [#id1, #id1],
                    iterator_types = ["parallel"]}
      ins(%z : memref<?xf32>) outs(%x : memref<?xf32>) {
    ^bb0(%zv: f32, %out: f32):
      %sum = arith.addf %out, %zv : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      indexing_maps = [#mat, #col, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : memref<?x?xf32, strided<[?, 1], offset: ?>>,
                   memref<?xf32>) outs(%w : memref<?xf32>) {
    ^bb0(%av: f32, %xv: f32, %out: f32):
      %scaled = arith.mulf %alpha, %av : f32
      %product = arith.mulf %scaled, %xv : f32
      %sum = arith.addf %out, %product : f32
      linalg.yield %sum : f32
    }
    return
  }
}
