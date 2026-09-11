// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

module {
  // This deliberately has the old sort recognizer's name, signature, while,
  // and comparator fingerprint.  Those clues must never authorize a complete
  // CUB replacement again.
  func.func @aten_sort_cpu(%input: memref<?x64xf32>,
                           %values: memref<?x64xf32>,
                           %indices: memref<?x64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %x = affine.load %input[%c0, %c0] : memref<?x64xf32>
    %y = affine.load %input[%c0, %c1] : memref<?x64xf32>
    %less = arith.cmpf olt, %x, %y : f32
    %result = scf.while (%condition = %less) : (i1) -> i1 {
      scf.condition(%condition) %false : i1
    } do {
    ^bb0(%condition: i1):
      scf.yield %condition : i1
    }
    return
  }
}

// CHECK-LABEL: func.func @aten_sort_cpu
// CHECK: scf.while
// CHECK-NOT: kernel.launch @cubSegmented{{Sort}}Descending_f32_i32_memref
