// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s 2>&1 | sed '/^\/\/ CHECK/d' | FileCheck %s

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

module {
  func.func @dot(%x: memref<?xf64>, %y: memref<?xf64>,
                 %storage: memref<f64>, %n: index) {
    %out = polygeist.submap(%storage, %n) {map = #scalar}
        : (memref<f64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["reduction"]} ins(%x, %y : memref<?xf64>, memref<?xf64>)
      outs(%out : memref<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out_value: f64):
      %product = arith.mulf %in, %in_0 : f64
      %sum = arith.addf %out_value, %product : f64
      linalg.yield %sum : f64
    }
    return
  }
}

// CHECK: kernel.launch @cublasDdot_memref
// CHECK-NOT: linalg.generic
