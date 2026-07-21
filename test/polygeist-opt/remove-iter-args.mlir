// RUN: polygeist-opt --remove-iter-args --split-input-file %s | FileCheck %s

// ============================================================================
// AFFINE.FOR TEST CASES
// ============================================================================

// Test case 1: Simple direct store (should work with original implementation)
// CHECK-LABEL: func.func @test_direct_store
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[VAL]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
// CHECK-NOT: affine.yield {{.*}} : f64
func.func @test_direct_store(%A: memref<?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  %result_mem = memref.alloc() : memref<f64>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  affine.store %sum, %result_mem[] : memref<f64>
  
  return
}

// -----

// Test case 2: Multiply after reduction (distributivity)
// Pattern: result = alpha * sum → sum = acc + (alpha * value)
// CHECK-LABEL: func.func @test_multiply_after_add
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK: %[[PROD:.*]] = arith.mulf %{{.*}}, %[[VAL]]
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[PROD]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
func.func @test_multiply_after_add(%A: memref<?xf64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  %result_mem = memref.alloc() : memref<f64>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  %scaled = arith.mulf %alpha, %sum : f64
  affine.store %scaled, %result_mem[] : memref<f64>
  
  return
}

// -----

// Test case 3: Addition with loop-invariant load (init adjustment)
// Pattern: result = C + sum → init = C, then direct store
// CHECK-LABEL: func.func @test_add_with_invariant_load
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[VAL]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
// CHECK-NOT: affine.load %{{.*}}[] : memref<f64>
func.func @test_add_with_invariant_load(%A: memref<?xf64>, %C: memref<f64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  %old_c = affine.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %sum : f64
  affine.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 4: Full GEMM pattern (multiply + add with load)
// Pattern: C = C + alpha * sum (most complex case)
// CHECK-LABEL: func.func @test_gemm_pattern
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK: %[[PROD:.*]] = arith.mulf %{{.*}}, %[[VAL]]
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[PROD]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
func.func @test_gemm_pattern(%A: memref<?xf64>, %C: memref<f64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %sum : f64
  %old_c = affine.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %scaled : f64
  affine.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 5: Realistic GEMM inner loop
// C[i,j] += alpha * sum_k(A[i,k] * B[k,j])
// CHECK-LABEL: func.func @test_gemm_inner_loop
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[C_LOADED:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: %[[A_VAL:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: %[[B_VAL:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: %[[PROD1:.*]] = arith.mulf %[[A_VAL]], %[[B_VAL]]
// CHECK: %[[PROD2:.*]] = arith.mulf %{{.*}}, %[[PROD1]]
// CHECK: %[[SUM:.*]] = arith.addf %[[C_LOADED]], %[[PROD2]]
// CHECK: affine.store %[[SUM]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
func.func @test_gemm_inner_loop(
    %A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>,
    %i: index, %j: index, %K: index, %lda: index, %ldb: index, %ldc: index,
    %alpha: f64) {
  %c0 = arith.constant 0 : index
  %init = arith.constant 0.0 : f64
  
  %dot_product = affine.for %k = %c0 to %K iter_args(%acc = %init) -> (f64) {
    %a_ik = affine.load %A[%i, %k] : memref<?x?xf64>
    %b_kj = affine.load %B[%k, %j] : memref<?x?xf64>
    %prod = arith.mulf %a_ik, %b_kj : f64
    %new_acc = arith.addf %acc, %prod : f64
    affine.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %dot_product : f64
  %old_c = affine.load %C[%i, %j] : memref<?x?xf64>
  %new_c = arith.addf %old_c, %scaled : f64
  affine.store %new_c, %C[%i, %j] : memref<?x?xf64>
  
  return
}

// -----

// Test case 6: Multiply-reduction with a post-loop scale.
// Distributivity does NOT apply (yield isn't addition), so the fast path bails.
// The alloca fallback handles it: one slot for the product accumulator, the
// post-loop scale runs after the final load.
// CHECK-LABEL: func.func @test_multiply_after_multiply
// CHECK-NOT: iter_args
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<f64>
// CHECK: affine.store %{{.*}}, %[[SLOT]][] : memref<f64>
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK: arith.mulf %[[ACC]], %{{.*}} : f64
// CHECK: affine.store %{{.*}}, %[[SLOT]][] : memref<f64>
// CHECK: }
// CHECK: %[[FIN:.*]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK: arith.mulf %{{.*}}, %[[FIN]] : f64
func.func @test_multiply_after_multiply(%A: memref<?xf64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 1.0 : f64
  %result_mem = memref.alloc() : memref<f64>
  
  %product = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.mulf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  %scaled = arith.mulf %alpha, %product : f64
  affine.store %scaled, %result_mem[] : memref<f64>
  
  return
}

// -----

// Test case 7: Multiple uses of the loop result.
// The fast path's hasOneUse() guard rejects this. The alloca fallback handles
// it by RAUWing the old result with a single post-loop load that both stores
// then consume.
// CHECK-LABEL: func.func @test_multiple_uses
// CHECK-NOT: iter_args
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<f64>
// CHECK: affine.for
// CHECK: }
// CHECK: %[[FIN:.*]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK: affine.store %[[FIN]], %{{.*}}[] : memref<f64>
// CHECK: affine.store %[[FIN]], %{{.*}}[] : memref<f64>
func.func @test_multiple_uses(%A: memref<?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  %result1 = memref.alloc() : memref<f64>
  %result2 = memref.alloc() : memref<f64>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  
  affine.store %sum, %result1[] : memref<f64>
  affine.store %sum, %result2[] : memref<f64>
  
  return
}

// -----

// ============================================================================
// INTEGER TESTS (AFFINE)
// ============================================================================

// Test case 8: Integer addition - direct store
// CHECK-LABEL: func.func @test_integer_direct_store
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_direct_store(%A: memref<?xi32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %result_mem = memref.alloc() : memref<i32>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  affine.store %sum, %result_mem[] : memref<i32>
  
  return
}

// -----

// Test case 9: Integer multiply after reduction
// CHECK-LABEL: func.func @test_integer_multiply_after_add
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_multiply_after_add(%A: memref<?xi32>, %n: index, %alpha: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %result_mem = memref.alloc() : memref<i32>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  %scaled = arith.muli %alpha, %sum : i32
  affine.store %scaled, %result_mem[] : memref<i32>
  
  return
}

// -----

// Test case 10: Integer addition with loop-invariant load
// CHECK-LABEL: func.func @test_integer_add_with_invariant_load
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_add_with_invariant_load(%A: memref<?xi32>, %C: memref<i32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  %old_c = affine.load %C[] : memref<i32>
  %new_c = arith.addi %old_c, %sum : i32
  affine.store %new_c, %C[] : memref<i32>
  
  return
}

// -----

// Test case 11: Full integer GEMM-like pattern
// CHECK-LABEL: func.func @test_integer_gemm_pattern
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_gemm_pattern(%A: memref<?xi32>, %C: memref<i32>, %n: index, %alpha: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  
  %scaled = arith.muli %alpha, %sum : i32
  %old_c = affine.load %C[] : memref<i32>
  %new_c = arith.addi %old_c, %scaled : i32
  affine.store %new_c, %C[] : memref<i32>
  
  return
}

// -----

// Test case 12: Integer matrix multiply inner loop
// CHECK-LABEL: func.func @test_integer_gemm_inner_loop
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xi32>
// CHECK: arith.muli
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_gemm_inner_loop(
    %A: memref<?x?xi32>, %B: memref<?x?xi32>, %C: memref<?x?xi32>,
    %i: index, %j: index, %K: index, %lda: index, %ldb: index, %ldc: index,
    %alpha: i32) {
  %c0 = arith.constant 0 : index
  %init = arith.constant 0 : i32
  
  %dot_product = affine.for %k = %c0 to %K iter_args(%acc = %init) -> (i32) {
    %a_ik = affine.load %A[%i, %k] : memref<?x?xi32>
    %b_kj = affine.load %B[%k, %j] : memref<?x?xi32>
    %prod = arith.muli %a_ik, %b_kj : i32
    %new_acc = arith.addi %acc, %prod : i32
    affine.yield %new_acc : i32
  }
  
  %scaled = arith.muli %alpha, %dot_product : i32
  %old_c = affine.load %C[%i, %j] : memref<?x?xi32>
  %new_c = arith.addi %old_c, %scaled : i32
  affine.store %new_c, %C[%i, %j] : memref<?x?xi32>
  
  return
}

// -----

// ============================================================================
// SCF.FOR TEST CASES
// ============================================================================

// Test case 13: SCF simple direct store
// CHECK-LABEL: func.func @test_scf_direct_store
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_direct_store(%A: memref<?xf64>, %result: memref<f64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  memref.store %sum, %result[] : memref<f64>
  
  return
}

// -----

// Test case 14: SCF multiply after loop
// CHECK-LABEL: func.func @test_scf_multiply_after
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_multiply_after(%A: memref<?xf64>, %C: memref<f64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %sum : f64
  memref.store %scaled, %C[] : memref<f64>
  
  return
}

// -----

// Test case 15: SCF add with invariant load
// CHECK-LABEL: func.func @test_scf_add_with_load
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_add_with_load(%A: memref<?xf64>, %C: memref<f64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  
  %old_c = memref.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %sum : f64
  memref.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 16: SCF full GEMM pattern
// CHECK-LABEL: func.func @test_scf_gemm_pattern
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_gemm_pattern(%A: memref<?xf64>, %C: memref<f64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %sum : f64
  %old_c = memref.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %scaled : f64
  memref.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 17: SCF integer operations
// CHECK-LABEL: func.func @test_scf_integer_gemm
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<i32>
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
func.func @test_scf_integer_gemm(%A: memref<?xi32>, %C: memref<i32>, %n: index, %alpha: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %val = memref.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    scf.yield %new_acc : i32
  }
  
  %scaled = arith.muli %alpha, %sum : i32
  %old_c = memref.load %C[] : memref<i32>
  %new_c = arith.addi %old_c, %scaled : i32
  memref.store %new_c, %C[] : memref<i32>

  return
}

// -----

// ============================================================================
// SURVEY-DERIVED CASES (alloca fallback)
// ============================================================================

// Survey r01: scalar reduction returned directly. Alloca path; final load
// becomes the return value.
// CHECK-LABEL: func.func @ddot
// CHECK:         %[[CST:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:         %[[SLOT:.+]] = memref.alloca() : memref<f64>
// CHECK:         affine.store %[[CST]], %[[SLOT]][] : memref<f64>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK:           %[[ACC:.+]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK:           %[[NEW:.+]] = arith.addf %[[ACC]], {{.*}} : f64
// CHECK:           affine.store %[[NEW]], %[[SLOT]][] : memref<f64>
// CHECK:         }
// CHECK:         %[[RES:.+]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK:         return %[[RES]] : f64
func.func @ddot(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %s = affine.for %i = 0 to %n iter_args(%acc = %cst) -> (f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %b = affine.load %y[%i] : memref<?xf64>
    %p = arith.mulf %a, %b : f64
    %new = arith.addf %acc, %p : f64
    affine.yield %new : f64
  }
  return %s : f64
}

// -----

// Survey r02: pure unary op (math.sqrt) sits between loop result and return.
// Alloca path: sqrt consumes the post-loop load.
// CHECK-LABEL: func.func @dnrm2
// CHECK:         %[[SLOT:.+]] = memref.alloca() : memref<f64>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK:         }
// CHECK:         %[[FIN:.+]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK:         %[[SQ:.+]] = math.sqrt %[[FIN]] : f64
// CHECK:         return %[[SQ]] : f64
func.func @dnrm2(%n: index, %x: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %s = affine.for %i = 0 to %n iter_args(%acc = %cst) -> (f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %p = arith.mulf %a, %a : f64
    %new = arith.addf %acc, %p : f64
    affine.yield %new : f64
  }
  %r = math.sqrt %s : f64
  return %r : f64
}

// -----

// Survey r06: loop result passed to a call. Alloca path; call argument is
// the post-loop load.
// CHECK-LABEL: func.func @log_sum
// CHECK:         %[[SLOT:.+]] = memref.alloca() : memref<f64>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK:         }
// CHECK:         %[[FIN:.+]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK:         call @sink(%[[FIN]]) : (f64) -> ()
// CHECK:         return
func.func @log_sum(%n: index, %x: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %s = affine.for %i = 0 to %n iter_args(%acc = %cst) -> (f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %new = arith.addf %acc, %a : f64
    affine.yield %new : f64
  }
  func.call @sink(%s) : (f64) -> ()
  return
}
func.func private @sink(f64)

// -----

// Survey r08: multi-iter_arg loop. The existing fast path bails (multi-iter
// guard); the alloca fallback creates one slot per iter_arg.
// CHECK-LABEL: func.func @two_reductions
// CHECK-DAG:     %[[S0:.+]] = memref.alloca() : memref<f64>
// CHECK-DAG:     %[[S1:.+]] = memref.alloca() : memref<f64>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK-DAG:       affine.load %[[S0]][] : memref<f64>
// CHECK-DAG:       affine.load %[[S1]][] : memref<f64>
// CHECK-DAG:       affine.store %{{.*}}, %[[S0]][] : memref<f64>
// CHECK-DAG:       affine.store %{{.*}}, %[[S1]][] : memref<f64>
// CHECK:         }
// CHECK-DAG:     affine.load %[[S0]][] : memref<f64>
// CHECK-DAG:     affine.load %[[S1]][] : memref<f64>
// CHECK:         return
func.func @two_reductions(%n: index, %x: memref<?xf64>,
                          %m: memref<?xf64>, %q: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %r:2 = affine.for %i = 0 to %n
        iter_args(%s = %cst, %ss = %cst) -> (f64, f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %ns = arith.addf %s, %a : f64
    %sq = arith.mulf %a, %a : f64
    %nss = arith.addf %ss, %sq : f64
    affine.yield %ns, %nss : f64, f64
  }
  affine.store %r#0, %m[0] : memref<?xf64>
  affine.store %r#1, %q[0] : memref<?xf64>
  return
}

// -----

// Survey r11: product reduction (mulf accumulator). Alloca path is operator-
// agnostic — the body is cloned verbatim.
// CHECK-LABEL: func.func @prod
// CHECK:         %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// CHECK:         %[[SLOT:.+]] = memref.alloca() : memref<f64>
// CHECK:         affine.store %[[ONE]], %[[SLOT]][] : memref<f64>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK:           %[[ACC:.+]] = affine.load %[[SLOT]][] : memref<f64>
// CHECK:           arith.mulf %[[ACC]], {{.*}} : f64
// CHECK:           affine.store %{{.*}}, %[[SLOT]][] : memref<f64>
// CHECK:         }
// CHECK:         affine.load %[[SLOT]][] : memref<f64>
// CHECK:         return
func.func @prod(%n: index, %x: memref<?xf64>) -> f64 {
  %one = arith.constant 1.000000e+00 : f64
  %p = affine.for %i = 0 to %n iter_args(%acc = %one) -> (f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %new = arith.mulf %acc, %a : f64
    affine.yield %new : f64
  }
  return %p : f64
}

// -----

// Survey r14: integer-typed iter_arg, post-loop result cast to index and used
// as an affine.for upper bound. RAUW propagates through the cast naturally.
// CHECK-LABEL: func.func @hist
// CHECK:         %[[SLOT:.+]] = memref.alloca() : memref<i32>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK:         }
// CHECK:         %[[FIN:.+]] = affine.load %[[SLOT]][] : memref<i32>
// CHECK:         %[[FINI:.+]] = arith.index_cast %[[FIN]] : i32 to index
// CHECK:         affine.for {{.*}} = 0 to %[[FINI]]
func.func @hist(%n: index, %x: memref<?xf64>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %count = affine.for %i = 0 to %n iter_args(%c = %c0) -> (i32) {
    %a = affine.load %x[%i] : memref<?xf64>
    %p = arith.cmpf ogt, %a, %cst : f64
    %nc = scf.if %p -> (i32) {
      %inc = arith.addi %c, %c1 : i32
      scf.yield %inc : i32
    } else {
      scf.yield %c : i32
    }
    affine.yield %nc : i32
  }
  %ci = arith.index_cast %count : i32 to index
  affine.for %j = 0 to %ci {
    %ji = arith.index_cast %j : index to i32
    func.call @use_int(%ji) : (i32) -> ()
  }
  return
}
func.func private @use_int(i32)

// -----

// Nested reductions (survey r15): inner iter_arg's result feeds the outer
// iter_arg's body. Both loops should be rewritten — inner first by the
// greedy driver, then outer.
// CHECK-LABEL: func.func @dist
// CHECK:         %[[OUT:.+]] = memref.alloca() : memref<f64>
// CHECK:         affine.for {{.*}} {
// CHECK-NOT:       iter_args
// CHECK:           %[[IN:.+]] = memref.alloca() : memref<f64>
// CHECK:           affine.for {{.*}} {
// CHECK-NOT:         iter_args
// CHECK:             affine.load %[[IN]][] : memref<f64>
// CHECK:             affine.store %{{.*}}, %[[IN]][] : memref<f64>
// CHECK:           }
// CHECK:           affine.load %[[IN]][] : memref<f64>
// CHECK:           affine.store %{{.*}}, %[[OUT]][] : memref<f64>
// CHECK:         }
// CHECK:         %[[RES:.+]] = affine.load %[[OUT]][] : memref<f64>
// CHECK:         return %[[RES]] : f64
func.func @dist(%m: index, %n: index, %A: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %total = affine.for %i = 0 to %m iter_args(%t = %cst) -> (f64) {
    %row = affine.for %j = 0 to %n iter_args(%r = %cst) -> (f64) {
      %v = affine.load %A[%i * symbol(%n) + %j] : memref<?xf64>
      %nr = arith.addf %r, %v : f64
      affine.yield %nr : f64
    }
    %sq = arith.mulf %row, %row : f64
    %nt = arith.addf %t, %sq : f64
    affine.yield %nt : f64
  }
  return %total : f64
}

// -----

// The unchanged load/store cleanup must not cross an intervening write. The
// final store restores the value loaded before that write and is observable.
// CHECK-LABEL: func.func @preserve_restore_after_write
// CHECK: %[[OLD:.+]] = affine.load %[[A:.+]][0] : memref<?xf64>
// CHECK: affine.store %{{.+}}, %[[A]][0] : memref<?xf64>
// CHECK: affine.store %[[OLD]], %[[A]][0] : memref<?xf64>
func.func @preserve_restore_after_write(%a: memref<?xf64>, %replacement: f64) {
  %old = affine.load %a[0] : memref<?xf64>
  affine.store %replacement, %a[0] : memref<?xf64>
  affine.store %old, %a[0] : memref<?xf64>
  return
}
