// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run --show-structured-regions 2>&1 | FileCheck %s

#id1 = affine_map<(d0) -> (d0)>
#left = affine_map<(d0)[s0] -> (d0 + s0 - 1)>
#center = affine_map<(d0)[s0] -> (d0 + s0)>
#right = affine_map<(d0)[s0] -> (d0 + s0 + 1)>

module {
  // A plain reduction is represented and classified even without an outer
  // affine/scf loop.  Algebraic operand order does not affect the class.
  func.func @sum(%x: memref<?xf64>, %out: memref<f64>) {
    linalg.generic {
        indexing_maps = [affine_map<(k) -> (k)>, affine_map<(k) -> ()>],
        iterator_types = ["reduction"]}
        ins(%x : memref<?xf64>) outs(%out : memref<f64>) {
    ^bb0(%in: f64, %acc: f64):
      %next = arith.addf %in, %acc : f64
      linalg.yield %next : f64
    }
    return
  }

  // Three distinct affine views of one root encode a neighbourhood rather
  // than three unrelated pointwise inputs.
  func.func @stencil(%x: memref<?xf64>, %out: memref<?xf64>, %offset: index) {
    %xm = polygeist.submap(%x, %offset) {map = #left}
        : (memref<?xf64>, index) -> memref<?xf64>
    %xc = polygeist.submap(%x, %offset) {map = #center}
        : (memref<?xf64>, index) -> memref<?xf64>
    %xp = polygeist.submap(%x, %offset) {map = #right}
        : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#id1, #id1, #id1, #id1],
                    iterator_types = ["parallel"]}
        ins(%xm, %xc, %xp : memref<?xf64>, memref<?xf64>, memref<?xf64>)
        outs(%out : memref<?xf64>) {
    ^bb0(%a: f64, %b: f64, %c: f64, %old: f64):
      %ab = arith.addf %a, %b : f64
      %abc = arith.addf %ab, %c : f64
      linalg.yield %abc : f64
    }
    return
  }

  func.func @gemm(%a: memref<?x?xf64>, %b: memref<?x?xf64>,
                  %c: memref<?x?xf64>) {
    linalg.generic {
        indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                         affine_map<(i, j, k) -> (k, j)>,
                         affine_map<(i, j, k) -> (i, j)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : memref<?x?xf64>, memref<?x?xf64>)
        outs(%c : memref<?x?xf64>) {
    ^bb0(%av: f64, %bv: f64, %cv: f64):
      %product = arith.mulf %bv, %av : f64
      %next = arith.addf %product, %cv : f64
      linalg.yield %next : f64
    }
    return
  }

  // This remains imperative because the destination index depends on input
  // data.  Report it as a candidate, never as an executable launch.
  func.func @histogram(%samples: tensor<?xf32>, %hist0: tensor<?xf32>)
      -> tensor<?xf32> {
    %result = affine.for %i = 0 to 1024 iter_args(%hist = %hist0)
        -> tensor<?xf32> {
      %sample = tensor.extract %samples[%i] : tensor<?xf32>
      %bin32 = arith.fptosi %sample : f32 to i32
      %bin = arith.index_cast %bin32 : i32 to index
      %old = tensor.extract %hist[%bin] : tensor<?xf32>
      %one = arith.constant 1.0 : f32
      %next = arith.addf %old, %one : f32
      %updated = tensor.insert %next into %hist[%bin] : tensor<?xf32>
      affine.yield %updated : tensor<?xf32>
    }
    return %result : tensor<?xf32>
  }

  // Canonical CSR row traversal: rowPtr bounds, column-indexed gather, and a
  // multiply-add reduction.  It is analysis-only until roles and ABI are
  // validated by a cuSPARSE lowering adapter.
  func.func @csr_spmv(%rowptr: memref<?xindex>, %col: memref<?xindex>,
                      %values: memref<?xf32>, %x: memref<?xf32>,
                      %y: memref<?xf32>, %rows: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    scf.for %row = %c0 to %rows step %c1 {
      %lo = memref.load %rowptr[%row] : memref<?xindex>
      %row1 = arith.addi %row, %c1 : index
      %hi = memref.load %rowptr[%row1] : memref<?xindex>
      %sum = scf.for %p = %lo to %hi step %c1
          iter_args(%acc = %zero) -> f32 {
        %column = memref.load %col[%p] : memref<?xindex>
        %value = memref.load %values[%p] : memref<?xf32>
        %xvalue = memref.load %x[%column] : memref<?xf32>
        %product = arith.mulf %value, %xvalue : f32
        %next = arith.addf %acc, %product : f32
        scf.yield %next : f32
      }
      memref.store %sum, %y[%row] : memref<?xf32>
    }
    return
  }
}

// CHECK: structured_fusion body#[0]
// CHECK-SAME: extracted=scalar_sum_reduction
// CHECK: structured_fusion body#[1]
// CHECK-SAME: extracted=affine_stencil
// CHECK: structured_fusion body#[2]
// CHECK-SAME: extracted=dense_gemm
// CHECK: residual_idiom_candidate body#[]
// CHECK-SAME: kind=csr_spmv
// CHECK-SAME: lowering_blocker=cuSPARSE route available after i32-index and f32/f64 operand validation
// CHECK: residual_idiom_candidate body#[]
// CHECK-SAME: kind=indirect_histogram
// CHECK-SAME: lowering_blocker=needs bin-range proof and atomic/collision-safe GPU lowering
