// RUN: polygeist-opt --delinearize-memref-accesses --split-input-file %s | FileCheck %s

// Row-major 2-D buffers linearized by hand (C[i*n + j]); the scale and the
// accumulate statements both go through the same view of C.
// CHECK-LABEL: func.func @gemm_flat
// CHECK-SAME: (%[[NI:.*]]: index, %[[NJ:.*]]: index, %[[NK:.*]]: index, %{{.*}}: f64, %{{.*}}: f64, %[[C:.*]]: memref<?xf64>, %[[A:.*]]: memref<?xf64>, %[[B:.*]]: memref<?xf64>)
// CHECK: %[[CR:.*]] = affine.max #{{.*}}()[%[[NI]]]
// CHECK: %[[CV:.*]] = memref.reinterpret_cast %[[C]] to offset: [0], sizes: [%[[CR]], %[[NJ]]], strides: [%[[NJ]], 1] : memref<?xf64> to memref<?x?xf64>
// CHECK: %[[AV:.*]] = memref.reinterpret_cast %[[A]] to offset: [0], sizes: [%{{.*}}, %[[NK]]], strides: [%[[NK]], 1]
// CHECK: %[[BV:.*]] = memref.reinterpret_cast %[[B]] to offset: [0], sizes: [%{{.*}}, %[[NJ]]], strides: [%[[NJ]], 1]
// CHECK: affine.for %[[I:.*]] = 0 to %[[NI]]
// CHECK: affine.load %[[CV]][%[[I]], %[[J:.*]]] : memref<?x?xf64>
// CHECK: affine.store %{{.*}}, %[[CV]][%[[I]], %[[J]]]
// CHECK: affine.load %[[AV]][%[[I]], %[[K:.*]]] : memref<?x?xf64>
// CHECK: affine.load %[[BV]][%[[K]], %[[J2:.*]]] : memref<?x?xf64>
// CHECK: affine.store %{{.*}}, %[[CV]][%[[I]], %[[J2]]]
func.func @gemm_flat(%ni: index, %nj: index, %nk: index, %alpha: f64, %beta: f64,
                     %C: memref<?xf64>, %A: memref<?xf64>, %B: memref<?xf64>) {
  affine.for %i = 0 to %ni {
    affine.for %j = 0 to %nj {
      %0 = affine.load %C[%j + %i * symbol(%nj)] : memref<?xf64>
      %1 = arith.mulf %0, %beta : f64
      affine.store %1, %C[%j + %i * symbol(%nj)] : memref<?xf64>
    }
    affine.for %k = 0 to %nk {
      affine.for %j = 0 to %nj {
        %0 = affine.load %A[%k + %i * symbol(%nk)] : memref<?xf64>
        %1 = arith.mulf %alpha, %0 : f64
        %2 = affine.load %B[%j + %k * symbol(%nj)] : memref<?xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %C[%j + %i * symbol(%nj)] : memref<?xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %C[%j + %i * symbol(%nj)] : memref<?xf64>
      }
    }
  }
  return
}

// -----

// 3-D: in[(i*N + j)*M + k] -> view [L x N x M] with strides [N*M, M, 1].
// CHECK-LABEL: func.func @scale3d
// CHECK-SAME: (%[[L:.*]]: index, %[[N:.*]]: index, %[[M:.*]]: index, %[[OUT:.*]]: memref<?xf64>, %[[IN:.*]]: memref<?xf64>)
// CHECK: %[[S0:.*]] = arith.muli %[[N]], %[[M]] : index
// CHECK: %[[OV:.*]] = memref.reinterpret_cast %[[OUT]] to offset: [0], sizes: [%{{.*}}, %[[N]], %[[M]]], strides: [%[[S0]], %[[M]], 1] : memref<?xf64> to memref<?x?x?xf64>
// CHECK: %[[S1:.*]] = arith.muli %[[N]], %[[M]] : index
// CHECK: %[[IV:.*]] = memref.reinterpret_cast %[[IN]] to offset: [0], sizes: [%{{.*}}, %[[N]], %[[M]]], strides: [%[[S1]], %[[M]], 1] : memref<?xf64> to memref<?x?x?xf64>
// CHECK: affine.load %[[IV]][%[[I:.*]], %[[J:.*]], %[[K:.*]]] : memref<?x?x?xf64>
// CHECK: affine.store %{{.*}}, %[[OV]][%[[I]], %[[J]], %[[K]]] : memref<?x?x?xf64>
func.func @scale3d(%l: index, %n: index, %m: index, %out: memref<?xf64>, %in: memref<?xf64>) {
  %cst = arith.constant 2.0 : f64
  affine.for %i = 0 to %l {
    affine.for %j = 0 to %n {
      affine.for %k = 0 to %m {
        %0 = affine.load %in[%k + (%j + %i * symbol(%n)) * symbol(%m)] : memref<?xf64>
        %1 = arith.mulf %0, %cst : f64
        affine.store %1, %out[%k + (%j + %i * symbol(%n)) * symbol(%m)] : memref<?xf64>
      }
    }
  }
  return
}

// -----

// Stencil offsets: +-1 stays in the column, +-n moves the row; the interior
// bounds 1 <= i, j < n-1 prove every column index in [0, n).
// CHECK-LABEL: func.func @jacobi2d_flat
// CHECK: %[[AV:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [%{{.*}}, %[[N:.*]]], strides: [%[[N]], 1]
// CHECK: affine.for %[[I:.*]] = 1 to
// CHECK: affine.for %[[J:.*]] = 1 to
// CHECK: affine.load %[[AV]][%[[I]], %[[J]]]
// CHECK: affine.load %[[AV]][%[[I]], %[[J]] - 1]
// CHECK: affine.load %[[AV]][%[[I]], %[[J]] + 1]
// CHECK: affine.load %[[AV]][%[[I]] + 1, %[[J]]]
// CHECK: affine.load %[[AV]][%[[I]] - 1, %[[J]]]
#ub = affine_map<()[s0] -> (s0 - 1)>
func.func @jacobi2d_flat(%n: index, %A: memref<?xf64>, %B: memref<?xf64>) {
  affine.for %i = 1 to #ub()[%n] {
    affine.for %j = 1 to #ub()[%n] {
      %1 = affine.load %A[%j + %i * symbol(%n)] : memref<?xf64>
      %2 = affine.load %A[%j + %i * symbol(%n) - 1] : memref<?xf64>
      %3 = affine.load %A[%j + %i * symbol(%n) + 1] : memref<?xf64>
      %4 = affine.load %A[%j + (%i + 1) * symbol(%n)] : memref<?xf64>
      %5 = affine.load %A[%j + (%i - 1) * symbol(%n)] : memref<?xf64>
      %6 = arith.addf %1, %2 : f64
      %7 = arith.addf %6, %3 : f64
      %8 = arith.addf %7, %4 : f64
      %9 = arith.addf %8, %5 : f64
      affine.store %9, %B[%j + %i * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// -----

// Constant strides: cgeist folds (i-1)*100 + j into i*100 + j - 100, which
// must become row i-1, while i*100 + j - 1 must stay column j-1.
// CHECK-LABEL: func.func @const_stencil
// CHECK: %[[AV:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [%{{.*}}, 100], strides: [100, 1] : memref<?xf64> to memref<?x100xf64>
// CHECK: affine.load %[[AV]][%[[I:.*]] - 1, %[[J:.*]]]
// CHECK: affine.load %[[AV]][%[[I]] + 1, %[[J]]]
// CHECK: affine.load %[[AV]][%[[I]], %[[J]] - 1]
// CHECK: affine.load %[[AV]][%[[I]], %[[J]] + 1]
func.func @const_stencil(%A: memref<?xf64>, %B: memref<?xf64>) {
  affine.for %i = 1 to 99 {
    affine.for %j = 1 to 99 {
      %0 = affine.load %A[%j + %i * 100 - 100] : memref<?xf64>
      %1 = affine.load %A[%j + %i * 100 + 100] : memref<?xf64>
      %2 = affine.load %A[%j + %i * 100 - 1] : memref<?xf64>
      %3 = affine.load %A[%j + %i * 100 + 1] : memref<?xf64>
      %4 = arith.addf %0, %1 : f64
      %5 = arith.addf %2, %3 : f64
      %6 = arith.addf %4, %5 : f64
      affine.store %6, %B[%j + %i * 100] : memref<?xf64>
    }
  }
  return
}

// -----

// Not rewritten: j < n says nothing about lda, so rows may overlap.
// CHECK-LABEL: func.func @gemv_lda
// CHECK-NOT: memref.reinterpret_cast
// CHECK: affine.load %{{.*}}[%{{.*}} + %{{.*}} * symbol(%{{.*}})] : memref<?xf64>
func.func @gemv_lda(%m: index, %n: index, %lda: index, %a: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
  affine.for %i = 0 to %m {
    affine.for %j = 0 to %n {
      %0 = affine.load %a[%j + %i * symbol(%lda)] : memref<?xf64>
      %1 = affine.load %x[%j] : memref<?xf64>
      %2 = arith.mulf %0, %1 : f64
      %3 = affine.load %y[%i] : memref<?xf64>
      %4 = arith.addf %3, %2 : f64
      affine.store %4, %y[%i] : memref<?xf64>
    }
  }
  return
}

// -----

// Not rewritten: a strided vector has no second dimension to recover, and
// incx >= 1 is not provable.
// CHECK-LABEL: func.func @strided1d
// CHECK-NOT: memref.reinterpret_cast
func.func @strided1d(%n: index, %incx: index, %a: f64, %x: memref<?xf64>) {
  affine.for %i = 0 to %n {
    %0 = affine.load %x[%i * symbol(%incx)] : memref<?xf64>
    %1 = arith.mulf %0, %a : f64
    affine.store %1, %x[%i * symbol(%incx)] : memref<?xf64>
  }
  return
}

// -----

// All-or-nothing: A is also walked flat (k < n*n), so only B gets a view.
// CHECK-LABEL: func.func @mixed_use
// CHECK-SAME: (%{{.*}}: index, %{{.*}}: index, %[[A:.*]]: memref<?xf64>, %[[B:.*]]: memref<?xf64>)
// CHECK-NOT: memref.reinterpret_cast %[[A]]
// CHECK: affine.store %{{.*}}, %[[A]][%{{[a-z0-9]+}}] : memref<?xf64>
// CHECK-NOT: memref.reinterpret_cast %[[A]]
// CHECK: %[[BV:.*]] = memref.reinterpret_cast %[[B]]
// CHECK-NOT: memref.reinterpret_cast %[[A]]
// CHECK: affine.load %[[BV]][%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: affine.load %[[A]][%{{.*}} + %{{.*}} * symbol(%{{.*}})] : memref<?xf64>
func.func @mixed_use(%n: index, %nn: index, %A: memref<?xf64>, %B: memref<?xf64>) {
  %cst = arith.constant 0.0 : f64
  affine.for %k = 0 to %nn {
    affine.store %cst, %A[%k] : memref<?xf64>
  }
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      %0 = affine.load %B[%j + %i * symbol(%n)] : memref<?xf64>
      %1 = affine.load %A[%j + %i * symbol(%n)] : memref<?xf64>
      %2 = arith.addf %1, %0 : f64
      affine.store %2, %A[%j + %i * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// -----

// Column-major BLAS operand a[i + j*lda] with i < m: the range i < lda comes
// from the enclosing argument check `if (lda >= MAX(1, m))` (OpenBLAS's
// interface returns early otherwise).
// CHECK-LABEL: func.func @gemv_guarded
// CHECK: %[[AV:.*]] = memref.reinterpret_cast
// CHECK: scf.if
// CHECK: affine.load %[[AV]][%{{.*}}, %{{.*}}] : memref<?x?xf64>
func.func @gemv_guarded(%m: i32, %n: i32, %lda: i32, %a: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %ldai = arith.index_cast %lda : i32 to index
  %ni = arith.index_cast %n : i32 to index
  %mi = arith.index_cast %m : i32 to index
  %gt = arith.cmpi sgt, %m, %c1 : i32
  %mx = arith.select %gt, %m, %c1 : i32
  %ok = arith.cmpi sge, %lda, %mx : i32
  scf.if %ok {
    affine.for %j = 0 to %ni {
      %xj = affine.load %x[%j] : memref<?xf64>
      affine.for %i = 0 to %mi {
        %aij = affine.load %a[%i + %j * symbol(%ldai)] : memref<?xf64>
        %p = arith.mulf %xj, %aij : f64
        %yi = affine.load %y[%i] : memref<?xf64>
        %s = arith.addf %yi, %p : f64
        affine.store %s, %y[%i] : memref<?xf64>
      }
    }
  }
  return
}

// -----

// The access runs where the check FAILED (lda < MAX(1, m)): no range proof.
// CHECK-LABEL: func.func @gemv_guard_else
// CHECK-NOT: memref.reinterpret_cast
func.func @gemv_guard_else(%m: i32, %n: i32, %lda: i32, %a: memref<?xf64>, %y: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %ldai = arith.index_cast %lda : i32 to index
  %ni = arith.index_cast %n : i32 to index
  %mi = arith.index_cast %m : i32 to index
  %gt = arith.cmpi sgt, %m, %c1 : i32
  %mx = arith.select %gt, %m, %c1 : i32
  %ok = arith.cmpi sge, %lda, %mx : i32
  scf.if %ok {
  } else {
    affine.for %j = 0 to %ni {
      affine.for %i = 0 to %mi {
        %aij = affine.load %a[%i + %j * symbol(%ldai)] : memref<?xf64>
        affine.store %aij, %y[%i] : memref<?xf64>
      }
    }
  }
  return
}

// -----

// A guard on an unrelated value proves nothing about i < lda.
// CHECK-LABEL: func.func @gemv_guard_unrelated
// CHECK-NOT: memref.reinterpret_cast
func.func @gemv_guard_unrelated(%m: i32, %n: i32, %lda: i32, %a: memref<?xf64>, %y: memref<?xf64>) {
  %ldai = arith.index_cast %lda : i32 to index
  %ni = arith.index_cast %n : i32 to index
  %mi = arith.index_cast %m : i32 to index
  %ok = arith.cmpi sge, %lda, %n : i32
  scf.if %ok {
    affine.for %j = 0 to %ni {
      affine.for %i = 0 to %mi {
        %aij = affine.load %a[%i + %j * symbol(%ldai)] : memref<?xf64>
        affine.store %aij, %y[%i] : memref<?xf64>
      }
    }
  }
  return
}

// -----

// The range fact comes from a dominating __builtin_assume (cgeist's form of
// the BLAS contract lda >= MAX(1, m)); the assumption is consumed.
// CHECK-LABEL: func.func @gemv_assume
// CHECK-NOT: __builtin_assume
// CHECK: %[[AV:.*]] = memref.reinterpret_cast
// CHECK: affine.load %[[AV]][%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK-NOT: __builtin_assume
func.func private @__builtin_assume(i8)
func.func @gemv_assume(%m: i32, %n: i32, %lda: i32, %a: memref<?xf64>, %y: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %gt = arith.cmpi sgt, %m, %c1 : i32
  %mx = arith.select %gt, %m, %c1 : i32
  %ok = arith.cmpi sge, %lda, %mx : i32
  %ok8 = arith.extui %ok : i1 to i8
  call @__builtin_assume(%ok8) : (i8) -> ()
  %ldai = arith.index_cast %lda : i32 to index
  %ni = arith.index_cast %n : i32 to index
  %mi = arith.index_cast %m : i32 to index
  affine.for %j = 0 to %ni {
    affine.for %i = 0 to %mi {
      %aij = affine.load %a[%i + %j * symbol(%ldai)] : memref<?xf64>
      affine.store %aij, %y[%i] : memref<?xf64>
    }
  }
  return
}
