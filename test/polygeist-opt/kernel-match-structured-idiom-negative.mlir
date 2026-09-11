// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run --show-structured-regions 2>&1 | FileCheck %s

#id = affine_map<(d0) -> (d0)>

module {
  // Reusing one direct pointwise input is not evidence of neighbour offsets.
  func.func @not_a_stencil(%x: memref<?xf32>, %out: memref<?xf32>) {
    linalg.generic {indexing_maps = [#id, #id, #id, #id],
                    iterator_types = ["parallel"]}
        ins(%x, %x, %x : memref<?xf32>, memref<?xf32>, memref<?xf32>)
        outs(%out : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %old: f32):
      %ab = arith.addf %a, %b : f32
      %abc = arith.addf %ab, %c : f32
      linalg.yield %abc : f32
    }
    return
  }

  // A fixed destination indexed by the induction variable is a map, not a
  // data-dependent histogram scatter.
  func.func @not_a_histogram(%values: tensor<?xf32>, %out0: tensor<?xf32>)
      -> tensor<?xf32> {
    %result = affine.for %i = 0 to 1024 iter_args(%out = %out0)
        -> tensor<?xf32> {
      %value = tensor.extract %values[%i] : tensor<?xf32>
      %old = tensor.extract %out[%i] : tensor<?xf32>
      %next = arith.addf %old, %value : f32
      %updated = tensor.insert %next into %out[%i] : tensor<?xf32>
      affine.yield %updated : tensor<?xf32>
    }
    return %result : tensor<?xf32>
  }

  // An indirect gather in a fixed-range loop is not sufficient to infer CSR:
  // CSR also requires two row-pointer loads defining the traversal bounds.
  func.func @not_csr(%col: memref<?xindex>, %values: memref<?xf32>,
                     %x: memref<?xf32>, %out: memref<f32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %sum = scf.for %p = %c0 to %n step %c1
        iter_args(%acc = %zero) -> f32 {
      %column = memref.load %col[%p] : memref<?xindex>
      %value = memref.load %values[%p] : memref<?xf32>
      %xvalue = memref.load %x[%column] : memref<?xf32>
      %product = arith.mulf %value, %xvalue : f32
      %next = arith.addf %acc, %product : f32
      scf.yield %next : f32
    }
    memref.store %sum, %out[] : memref<f32>
    return
  }
}

// CHECK-NOT: extracted=affine_stencil
// CHECK-NOT: kind=indirect_histogram
// CHECK-NOT: kind=csr_spmv
// CHECK: total: 0 matched / 1 bodies
