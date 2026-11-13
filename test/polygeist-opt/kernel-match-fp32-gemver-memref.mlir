// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^[[:space:]]*\/\/ \(CHECK\|SYNTAX\)/d' | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py --matcher-mode=syntactic %s | sed '/^[[:space:]]*\/\/ \(CHECK\|SYNTAX\)/d' | FileCheck %s --check-prefix=SYNTAX

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
  // CHECK: kernel.launch @cublasSgemv_alpha_memref({{.*}}%alpha)
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

  // A precision-specific ABI must never win solely because its scalar
  // expression and iterator structure match. This is the FP64 BiCG shape
  // that previously selected the FP32 SGEMV-alpha memref definition.
  // CHECK-LABEL: func.func @bicg_f64_no_alpha
  // CHECK-NOT: kernel.launch @cublasSgemv_alpha
  // CHECK: linalg.generic
  // SYNTAX-LABEL: func.func @bicg_f64_no_alpha
  // SYNTAX-NOT: kernel.launch @cublasSgemv_alpha
  // SYNTAX: linalg.generic
  func.func @bicg_f64_no_alpha(
      %a: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %x: memref<?xf64, strided<[1], offset: ?>>,
      %y: memref<?xf64>) {
    linalg.generic {
      indexing_maps = [#mat, #col, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : memref<?x?xf64, strided<[?, 1], offset: ?>>,
                   memref<?xf64, strided<[1], offset: ?>>)
      outs(%y : memref<?xf64>) {
    ^bb0(%av: f64, %xv: f64, %out: f64):
      %product = arith.mulf %av, %xv : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    }
    return
  }

  // If algebraic matching supplies an omitted alpha=1 for an otherwise legal
  // FP32 ABI, the required fourth launch operand must be materialized.
  // CHECK-LABEL: func.func @sgemv_f32_implicit_alpha
  // CHECK: [[ALPHA:%[a-zA-Z0-9_.$-]+]] = arith.constant 1.0 : f32
  // CHECK: kernel.launch @cublasSgemv_alpha_memref({{.*}}[[ALPHA]])
  func.func @sgemv_f32_implicit_alpha(
      %a: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %x: memref<?xf32, strided<[1], offset: ?>>,
      %y: memref<?xf32>) {
    linalg.generic {
      indexing_maps = [#mat, #col, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : memref<?x?xf32, strided<[?, 1], offset: ?>>,
                   memref<?xf32, strided<[1], offset: ?>>)
      outs(%y : memref<?xf32>) {
    ^bb0(%av: f32, %xv: f32, %out: f32):
      %product = arith.mulf %av, %xv : f32
      %sum = arith.addf %out, %product : f32
      linalg.yield %sum : f32
    }
    return
  }
}
