// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite 2>&1 | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  func.func @csr_spmv(%rowptr: memref<?xi32>, %col: memref<?xi32>,
                      %values: memref<?xf64>, %x: memref<?xf64>,
                      %y: memref<?xf64>, %rows: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    scf.for %row = %c0 to %rows step %c1 {
      %lo32 = memref.load %rowptr[%row] : memref<?xi32>
      %lo = arith.index_cast %lo32 : i32 to index
      %row1 = arith.addi %row, %c1 : index
      %hi32 = memref.load %rowptr[%row1] : memref<?xi32>
      %hi = arith.index_cast %hi32 : i32 to index
      %sum = scf.for %p = %lo to %hi step %c1
          iter_args(%acc = %zero) -> f64 {
        %column32 = memref.load %col[%p] : memref<?xi32>
        %column = arith.index_cast %column32 : i32 to index
        %value = memref.load %values[%p] : memref<?xf64>
        %xvalue = memref.load %x[%column] : memref<?xf64>
        %product = arith.mulf %value, %xvalue : f64
        %next = arith.addf %acc, %product : f64
        scf.yield %next : f64
      }
      memref.store %sum, %y[%row] : memref<?xf64>
    }
    return
  }
}

// y[row] = sum with a zero-initialized sum: y = 1*A*x + 0*y.
// CHECK: %[[ALPHA:cusparse_alpha_[0-9]+]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[BETA:cusparse_beta_[0-9]+]] = arith.constant 0.000000e+00 : f64
// CHECK: kernel.launch @cusparseSpMV_CSR_f64_memref
// CHECK-SAME: (%rows, %rowptr, %col, %values, %x, %y, %[[ALPHA]], %[[BETA]])
