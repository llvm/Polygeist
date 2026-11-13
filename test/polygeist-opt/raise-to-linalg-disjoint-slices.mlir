// RUN: polygeist-opt --raise-affine-to-linalg %s | FileCheck %s

module {
  func.func @constant_offset_slices(%input: memref<5xf64>,
                                    %output: memref<100xf64>) {
    affine.for %i = 0 to 5 {
      %value = affine.load %input[%i] : memref<5xf64>
      %root = math.sqrt %value : f64
      affine.store %root, %output[%i] : memref<100xf64>
      affine.store %root, %output[%i + 25] : memref<100xf64>
      affine.store %root, %output[%i + 50] : memref<100xf64>
      affine.store %root, %output[%i + 75] : memref<100xf64>
    }
    return
  }

  // The offset equals the iteration span, so the two address sets overlap at
  // output[4].  Keep this loop to guard against an unsound same-iteration-only
  // disjointness check.
  func.func @overlapping_offset(%input: memref<5xf64>,
                                %output: memref<10xf64>) {
    affine.for %i = 0 to 5 {
      %value = affine.load %input[%i] : memref<5xf64>
      affine.store %value, %output[%i] : memref<10xf64>
      affine.store %value, %output[%i + 4] : memref<10xf64>
    }
    return
  }
}

// CHECK-LABEL: func.func @constant_offset_slices
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK-SAME: outs({{.*}}, {{.*}}, {{.*}}, {{.*}}
// CHECK: math.sqrt
// CHECK: linalg.yield {{.*}}, {{.*}}, {{.*}}, {{.*}} : f64, f64, f64, f64

// CHECK-LABEL: func.func @overlapping_offset
// CHECK: affine.for
