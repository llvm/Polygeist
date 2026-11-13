// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite 2>&1 | sed '/^\/\/ CHECK/d' | FileCheck %s

// cuSPARSE computes y = alpha*A*x + beta*y.  alpha and beta are derived from
// how the row sum is written back; any other row update must stay as loops
// (previously y[r] += sum was emitted with the shim's fixed beta = 0).

#map = affine_map<(d0) -> (d0 + 1)>
module {
  // y[r] += sum  ->  alpha = 1, beta = 1
// CHECK-LABEL: func.func @accumulate(
// CHECK: %[[A:cusparse_alpha_[0-9]+]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[B:cusparse_beta_[0-9]+]] = arith.constant 1.000000e+00 : f64
// CHECK: kernel.launch @cusparseSpMV_CSR_f64_memref(%{{.*}}, %[[A]], %[[B]])
// CHECK-NOT: scf.for
  func.func @accumulate(%n: index, %rp: memref<?xi32>, %c: memref<?xi32>,
                        %v: memref<?xf64>, %x: memref<?xf64>,
                        %y: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    affine.for %r = 0 to %n {
      %lo32 = affine.load %rp[%r] : memref<?xi32>
      %hi32 = affine.load %rp[%r + 1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %lo = arith.index_cast %lo32 : i32 to index
      %sum = scf.for %k = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
        %col32 = memref.load %c[%k] : memref<?xi32>
        %col = arith.index_cast %col32 : i32 to index
        %xv = memref.load %x[%col] : memref<?xf64>
        %vv = memref.load %v[%k] : memref<?xf64>
        %p = arith.mulf %xv, %vv : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      %old = affine.load %y[%r] : memref<?xf64>
      %new = arith.addf %old, %sum : f64
      affine.store %new, %y[%r] : memref<?xf64>
    }
    return
  }

  // y[r] = a*sum + b*y[r]  ->  alpha = %a, beta = %b (the caller's scalars)
// CHECK-LABEL: func.func @scaled(
// CHECK: kernel.launch @cusparseSpMV_CSR_f64_memref(%n, %rp, %c, %v, %x, %y, %a, %b)
// CHECK-NOT: scf.for
  func.func @scaled(%n: index, %rp: memref<?xi32>, %c: memref<?xi32>,
                    %v: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>,
                    %a: f64, %b: f64) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    affine.for %r = 0 to %n {
      %lo32 = affine.load %rp[%r] : memref<?xi32>
      %hi32 = affine.load %rp[%r + 1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %lo = arith.index_cast %lo32 : i32 to index
      %sum = scf.for %k = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
        %col32 = memref.load %c[%k] : memref<?xi32>
        %col = arith.index_cast %col32 : i32 to index
        %xv = memref.load %x[%col] : memref<?xf64>
        %vv = memref.load %v[%k] : memref<?xf64>
        %p = arith.mulf %xv, %vv : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      %as = arith.mulf %a, %sum : f64
      %old = affine.load %y[%r] : memref<?xf64>
      %bo = arith.mulf %b, %old : f64
      %new = arith.addf %as, %bo : f64
      affine.store %new, %y[%r] : memref<?xf64>
    }
    return
  }

  // y[r] = y[r] - sum  ->  alpha = -1, beta = 1
// CHECK-LABEL: func.func @subtract(
// CHECK: %[[A2:cusparse_alpha_[0-9]+]] = arith.constant -1.000000e+00 : f64
// CHECK: %[[B2:cusparse_beta_[0-9]+]] = arith.constant 1.000000e+00 : f64
// CHECK: kernel.launch @cusparseSpMV_CSR_f64_memref(%{{.*}}, %[[A2]], %[[B2]])
  func.func @subtract(%n: index, %rp: memref<?xi32>, %c: memref<?xi32>,
                      %v: memref<?xf64>, %x: memref<?xf64>,
                      %y: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    affine.for %r = 0 to %n {
      %lo32 = affine.load %rp[%r] : memref<?xi32>
      %hi32 = affine.load %rp[%r + 1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %lo = arith.index_cast %lo32 : i32 to index
      %sum = scf.for %k = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
        %col32 = memref.load %c[%k] : memref<?xi32>
        %col = arith.index_cast %col32 : i32 to index
        %xv = memref.load %x[%col] : memref<?xf64>
        %vv = memref.load %v[%k] : memref<?xf64>
        %p = arith.mulf %xv, %vv : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      %old = affine.load %y[%r] : memref<?xf64>
      %new = arith.subf %old, %sum : f64
      affine.store %new, %y[%r] : memref<?xf64>
    }
    return
  }

  // y[r] = sum * sum is not a linear update: stays as loops.
// CHECK-LABEL: func.func @nonlinear(
// CHECK-NOT: kernel.launch
// CHECK: scf.for
  func.func @nonlinear(%n: index, %rp: memref<?xi32>, %c: memref<?xi32>,
                       %v: memref<?xf64>, %x: memref<?xf64>,
                       %y: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    affine.for %r = 0 to %n {
      %lo32 = affine.load %rp[%r] : memref<?xi32>
      %hi32 = affine.load %rp[%r + 1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %lo = arith.index_cast %lo32 : i32 to index
      %sum = scf.for %k = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
        %col32 = memref.load %c[%k] : memref<?xi32>
        %col = arith.index_cast %col32 : i32 to index
        %xv = memref.load %x[%col] : memref<?xf64>
        %vv = memref.load %v[%k] : memref<?xf64>
        %p = arith.mulf %xv, %vv : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      %sq = arith.mulf %sum, %sum : f64
      affine.store %sq, %y[%r] : memref<?xf64>
    }
    return
  }

  // A per-row scale (z[r] * sum) is not a cuSPARSE alpha: stays as loops.
// CHECK-LABEL: func.func @row_varying(
// CHECK-NOT: kernel.launch
// CHECK: scf.for
  func.func @row_varying(%n: index, %rp: memref<?xi32>, %c: memref<?xi32>,
                         %v: memref<?xf64>, %x: memref<?xf64>,
                         %y: memref<?xf64>, %z: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    affine.for %r = 0 to %n {
      %lo32 = affine.load %rp[%r] : memref<?xi32>
      %hi32 = affine.load %rp[%r + 1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %lo = arith.index_cast %lo32 : i32 to index
      %sum = scf.for %k = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
        %col32 = memref.load %c[%k] : memref<?xi32>
        %col = arith.index_cast %col32 : i32 to index
        %xv = memref.load %x[%col] : memref<?xf64>
        %vv = memref.load %v[%k] : memref<?xf64>
        %p = arith.mulf %xv, %vv : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      %zr = affine.load %z[%r] : memref<?xf64>
      %scaled = arith.mulf %zr, %sum : f64
      affine.store %scaled, %y[%r] : memref<?xf64>
    }
    return
  }

  // Storing the row sum anywhere else as well: stays as loops.
// CHECK-LABEL: func.func @extra_store(
// CHECK-NOT: kernel.launch
// CHECK: scf.for
  func.func @extra_store(%n: index, %rp: memref<?xi32>, %c: memref<?xi32>,
                         %v: memref<?xf64>, %x: memref<?xf64>,
                         %y: memref<?xf64>, %z: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    affine.for %r = 0 to %n {
      %lo32 = affine.load %rp[%r] : memref<?xi32>
      %hi32 = affine.load %rp[%r + 1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %lo = arith.index_cast %lo32 : i32 to index
      %sum = scf.for %k = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
        %col32 = memref.load %c[%k] : memref<?xi32>
        %col = arith.index_cast %col32 : i32 to index
        %xv = memref.load %x[%col] : memref<?xf64>
        %vv = memref.load %v[%k] : memref<?xf64>
        %p = arith.mulf %xv, %vv : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      affine.store %sum, %y[%r] : memref<?xf64>
      affine.store %sum, %z[%r] : memref<?xf64>
    }
    return
  }
}
