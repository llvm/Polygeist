// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

module {
  // Arrays of memref descriptors encode pointer indirection, not a dense
  // tensor.  Keep such loops explicit instead of constructing an invalid
  // linalg.submap whose indexing value escapes the loop that defines it.
  func.func @descriptor_array(%rows: memref<8xmemref<?xf32>>, %value: f32) {
    affine.for %i = 0 to 8 {
      %row = affine.load %rows[%i] : memref<8xmemref<?xf32>>
      affine.store %value, %row[0] : memref<?xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @descriptor_array
// CHECK: affine.for
// CHECK: affine.load {{.*}} : memref<8xmemref<?xf32>>
// CHECK: affine.store {{.*}} : memref<?xf32>
// CHECK-NOT: linalg.generic

// -----

module {
  // An indirect memref.load left inside the body is not a shaped Linalg
  // operand, so it cannot supply the iteration domain for a scalar result.
  func.func @indirect_scalar_reduction(%input: memref<?xf64>, %n: index)
      -> f64 {
    %zero = arith.constant 0.0 : f64
    %sum = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
      %value = memref.load %input[%i] : memref<?xf64>
      %next = arith.addf %acc, %value : f64
      affine.yield %next : f64
    }
    return %sum : f64
  }
}

// CHECK-LABEL: func.func @indirect_scalar_reduction
// CHECK: affine.for
// CHECK: memref.load
// CHECK-NOT: linalg.generic
