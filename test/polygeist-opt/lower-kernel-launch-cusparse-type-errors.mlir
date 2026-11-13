// RUN: polygeist-opt %s --split-input-file --lower-kernel-launch-to-cublas -verify-diagnostics

// These definitions deliberately use malformed signatures so the generic
// kernel.launch verifier accepts each launch. The ABI lowering must still
// reject it locally before querying dimensions or extracting data pointers.

module {
  kernel.defn @cusparseSpMV_CSR_f32_memref(
      %rows: index, %rowptr: memref<?xf32>, %cols: memref<?xi32>,
      %values: memref<?xf32>, %x: memref<?xf32>, %y: memref<?xf32>,
      %alpha: f32, %beta: f32) {
    kernel.yield
  }
  func.func @bad_spmv(%rows: index, %rowptr: memref<?xf32>,
                      %cols: memref<?xi32>, %values: memref<?xf32>,
                      %x: memref<?xf32>, %y: memref<?xf32>,
                      %alpha: f32, %beta: f32) {
    // expected-error@+1 {{cuSPARSE CSR SpMV: expected index rows, rank-1 i32 indices}}
    kernel.launch @cusparseSpMV_CSR_f32_memref(
        %rows, %rowptr, %cols, %values, %x, %y, %alpha, %beta) :
        (index, memref<?xf32>, memref<?xi32>, memref<?xf32>,
         memref<?xf32>, memref<?xf32>, f32, f32) -> ()
    return
  }
}

// -----

// alpha/beta must have the symbol's precision.
module {
  kernel.defn @cusparseSpMV_CSR_f32_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf32>, %x: memref<?xf32>, %y: memref<?xf32>,
      %alpha: f64, %beta: f64) {
    kernel.yield
  }
  func.func @bad_spmv_scalars(%rows: index, %rowptr: memref<?xi32>,
                              %cols: memref<?xi32>, %values: memref<?xf32>,
                              %x: memref<?xf32>, %y: memref<?xf32>,
                              %alpha: f64, %beta: f64) {
    // expected-error@+1 {{alpha/beta scalars matching the symbol precision}}
    kernel.launch @cusparseSpMV_CSR_f32_memref(
        %rows, %rowptr, %cols, %values, %x, %y, %alpha, %beta) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?xf32>, memref<?xf32>, f64, f64) -> ()
    return
  }
}

// -----

// The pre-alpha/beta six-operand form is rejected, not silently given beta=0.
module {
  kernel.defn @cusparseSpMV_CSR_f64_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
    kernel.yield
  }
  func.func @legacy_spmv(%rows: index, %rowptr: memref<?xi32>,
                         %cols: memref<?xi32>, %values: memref<?xf64>,
                         %x: memref<?xf64>, %y: memref<?xf64>) {
    // expected-error@+1 {{expected rows, row offsets, column indices, values, x, y, alpha, beta}}
    kernel.launch @cusparseSpMV_CSR_f64_memref(
        %rows, %rowptr, %cols, %values, %x, %y) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf64>,
         memref<?xf64>, memref<?xf64>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cudnnAdaptivePool_f32_flat2(
      %op: i32, %rank: i32, %n: i32, %c: i32, %i0: i32, %i1: i32,
      %i2: i32, %o0: i32, %o1: i32, %o2: i32,
      %input: memref<?xf64>, %output: memref<?xf32>) { kernel.yield }
  func.func @bad_adaptive_pool(
      %op: i32, %rank: i32, %n: i32, %c: i32, %i0: i32, %i1: i32,
      %i2: i32, %o0: i32, %o1: i32, %o2: i32,
      %input: memref<?xf64>, %output: memref<?xf32>) {
    // expected-error@+1 {{cudnnAdaptivePool_f32: trailing operands must be rank-1 shaped buffers}}
    kernel.launch @cudnnAdaptivePool_f32_flat2(
        %op, %rank, %n, %c, %i0, %i1, %i2, %o0, %o1, %o2,
        %input, %output) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
         memref<?xf64>, memref<?xf32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cudnnBatchNormBackward_f32_dx(
      %n: i32, %c: i32, %spatial: i32,
      %grad: memref<?x?x?xf32>, %x: memref<?x?x?x?xf32>,
      %mean: memref<?xf32>, %invstd: memref<?xf32>,
      %dx: memref<?x?x?x?xf32>) { kernel.yield }
  func.func @bad_batchnorm(
      %n: i32, %c: i32, %spatial: i32,
      %grad: memref<?x?x?xf32>, %x: memref<?x?x?x?xf32>,
      %mean: memref<?xf32>, %invstd: memref<?xf32>,
      %dx: memref<?x?x?x?xf32>) {
    // expected-error@+1 {{cudnnBatchNormBackward_f32: operand 3 must be a rank-4 f32 shaped buffer}}
    kernel.launch @cudnnBatchNormBackward_f32_dx(
        %n, %c, %spatial, %grad, %x, %mean, %invstd, %dx) :
        (i32, i32, i32, memref<?x?x?xf32>, memref<?x?x?x?xf32>,
         memref<?xf32>, memref<?xf32>, memref<?x?x?x?xf32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cubSegmentReduceLengths_f32_memref(
      %input: memref<?xf64>, %lengths: memref<?xi32>, %mode: i32,
      %output: memref<?xf32>) { kernel.yield }
  func.func @bad_segment_reduce(
      %input: memref<?xf64>, %lengths: memref<?xi32>, %mode: i32,
      %output: memref<?xf32>) {
    // expected-error@+1 {{length-segmented reduction expects rank-1 f32 input/output}}
    kernel.launch @cubSegmentReduceLengths_f32_memref(
        %input, %lengths, %mode, %output) :
        (memref<?xf64>, memref<?xi32>, i32, memref<?xf32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cusparseSpMM_CSR_f32_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf32>, %b: memref<?xf32>, %c: memref<?x?xf32>) {
    kernel.yield
  }
  func.func @bad_spmm(%rows: index, %rowptr: memref<?xi32>,
                      %cols: memref<?xi32>, %values: memref<?xf32>,
                      %b: memref<?xf32>, %c: memref<?x?xf32>) {
    // expected-error@+1 {{cuSPARSE SpMM: expected rank-1 i32 indices}}
    kernel.launch @cusparseSpMM_CSR_f32_memref(
        %rows, %rowptr, %cols, %values, %b, %c) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?xf32>, memref<?x?xf32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cusparseSpMM_BSR_f32_memref(
      %rows: index, %dim: index, %rowptr: memref<?xi32>,
      %cols: memref<?xi32>, %values: memref<?x?xf32>,
      %x: memref<?xf32>, %y: memref<?xf32>) { kernel.yield }
  func.func @bad_bsr(%rows: index, %dim: index,
                     %rowptr: memref<?xi32>, %cols: memref<?xi32>,
                     %values: memref<?x?xf32>, %x: memref<?xf32>,
                     %y: memref<?xf32>) {
    // expected-error@+1 {{cuSPARSE BSR SpMM: expected index dimensions}}
    kernel.launch @cusparseSpMM_BSR_f32_memref(
        %rows, %dim, %rowptr, %cols, %values, %x, %y) :
        (index, index, memref<?xi32>, memref<?xi32>, memref<?x?xf32>,
         memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cusparseSDDMM_CSR_f32_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %self: memref<?xf32>, %a: memref<?x?xf32>, %b: memref<?x?xf32>,
      %alpha: f64, %beta: f32, %out: memref<?xf32>) { kernel.yield }
  func.func @bad_sddmm(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %self: memref<?xf32>, %a: memref<?x?xf32>, %b: memref<?x?xf32>,
      %alpha: f64, %beta: f32, %out: memref<?xf32>) {
    // expected-error@+1 {{cuSPARSE CSR SDDMM: expected index rows}}
    kernel.launch @cusparseSDDMM_CSR_f32_memref(
        %rows, %rowptr, %cols, %self, %a, %b, %alpha, %beta, %out) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?x?xf32>, memref<?x?xf32>, f64, f32, memref<?xf32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cusparseXcoo2csr_i32_memref(
      %rows: index, %input: memref<?xi64>, %output: memref<?xi32>) {
    kernel.yield
  }
  func.func @bad_conversion(%rows: index, %input: memref<?xi64>,
                            %output: memref<?xi32>) {
    // expected-error@+1 {{cuSPARSE index conversion: expected index rows}}
    kernel.launch @cusparseXcoo2csr_i32_memref(%rows, %input, %output) :
        (index, memref<?xi64>, memref<?xi32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cusparseSpMV_JDS_f32_memref(
      %rows: index, %repetitions: index, %counts: memref<?xi32>,
      %offsets: memref<?xi32>, %columns: memref<?xi32>,
      %values: memref<?xf64>, %permutation: memref<?xi32>,
      %x: memref<?xf32>, %y: memref<?xf32>) { kernel.yield }
  func.func @bad_jds(
      %rows: index, %repetitions: index, %counts: memref<?xi32>,
      %offsets: memref<?xi32>, %columns: memref<?xi32>,
      %values: memref<?xf64>, %permutation: memref<?xi32>,
      %x: memref<?xf32>, %y: memref<?xf32>) {
    // expected-error@+1 {{cuSPARSE JDS SpMV: expected index rows/repetitions}}
    kernel.launch @cusparseSpMV_JDS_f32_memref(
        %rows, %repetitions, %counts, %offsets, %columns, %values,
        %permutation, %x, %y) :
        (index, index, memref<?xi32>, memref<?xi32>, memref<?xi32>,
         memref<?xf64>, memref<?xi32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
