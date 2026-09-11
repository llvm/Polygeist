// RUN: polygeist-opt %s --raise-affine-to-linalg | FileCheck %s

// Fully unrolled legacy code has no loop for the ordinary Linalg raiser to
// consume. Recover the iteration space from complete constant-index access
// patterns, without relying on symbol names.
module {
  func.func @arbitrary_matvec_name(%a: memref<?x2xf64>, %x: memref<?xf64>,
                                  %y: memref<?xf64>) {
    %y0 = affine.load %y[0] : memref<?xf64>
    %a00 = affine.load %a[0, 0] : memref<?x2xf64>
    %x0 = affine.load %x[0] : memref<?xf64>
    %p00 = arith.mulf %a00, %x0 : f64
    %s00 = arith.subf %y0, %p00 : f64
    %a10 = affine.load %a[1, 0] : memref<?x2xf64>
    %x1 = affine.load %x[1] : memref<?xf64>
    %p10 = arith.mulf %a10, %x1 : f64
    %s10 = arith.subf %s00, %p10 : f64
    affine.store %s10, %y[0] : memref<?xf64>
    %y1 = affine.load %y[1] : memref<?xf64>
    %a01 = affine.load %a[0, 1] : memref<?x2xf64>
    %x0b = affine.load %x[0] : memref<?xf64>
    %p01 = arith.mulf %a01, %x0b : f64
    %s01 = arith.subf %y1, %p01 : f64
    %a11 = affine.load %a[1, 1] : memref<?x2xf64>
    %x1b = affine.load %x[1] : memref<?xf64>
    %p11 = arith.mulf %a11, %x1b : f64
    %s11 = arith.subf %s01, %p11 : f64
    affine.store %s11, %y[1] : memref<?xf64>
    return
  }

  func.func @arbitrary_matmul_name(%a: memref<?x2xf64>, %b: memref<?x2xf64>,
                                  %c: memref<?x2xf64>) {
    %c00 = affine.load %c[0, 0] : memref<?x2xf64>
    %a00 = affine.load %a[0, 0] : memref<?x2xf64>
    %b00 = affine.load %b[0, 0] : memref<?x2xf64>
    %p000 = arith.mulf %a00, %b00 : f64
    %s000 = arith.subf %c00, %p000 : f64
    %a10 = affine.load %a[1, 0] : memref<?x2xf64>
    %b01 = affine.load %b[0, 1] : memref<?x2xf64>
    %p001 = arith.mulf %a10, %b01 : f64
    %s001 = arith.subf %s000, %p001 : f64
    affine.store %s001, %c[0, 0] : memref<?x2xf64>

    %c01 = affine.load %c[0, 1] : memref<?x2xf64>
    %a01 = affine.load %a[0, 1] : memref<?x2xf64>
    %b00b = affine.load %b[0, 0] : memref<?x2xf64>
    %p010 = arith.mulf %a01, %b00b : f64
    %s010 = arith.subf %c01, %p010 : f64
    %a11 = affine.load %a[1, 1] : memref<?x2xf64>
    %b01b = affine.load %b[0, 1] : memref<?x2xf64>
    %p011 = arith.mulf %a11, %b01b : f64
    %s011 = arith.subf %s010, %p011 : f64
    affine.store %s011, %c[0, 1] : memref<?x2xf64>

    %c10 = affine.load %c[1, 0] : memref<?x2xf64>
    %a00b = affine.load %a[0, 0] : memref<?x2xf64>
    %b10 = affine.load %b[1, 0] : memref<?x2xf64>
    %p100 = arith.mulf %a00b, %b10 : f64
    %s100 = arith.subf %c10, %p100 : f64
    %a10b = affine.load %a[1, 0] : memref<?x2xf64>
    %b11 = affine.load %b[1, 1] : memref<?x2xf64>
    %p101 = arith.mulf %a10b, %b11 : f64
    %s101 = arith.subf %s100, %p101 : f64
    affine.store %s101, %c[1, 0] : memref<?x2xf64>

    %c11 = affine.load %c[1, 1] : memref<?x2xf64>
    %a01b = affine.load %a[0, 1] : memref<?x2xf64>
    %b10b = affine.load %b[1, 0] : memref<?x2xf64>
    %p110 = arith.mulf %a01b, %b10b : f64
    %s110 = arith.subf %c11, %p110 : f64
    %a11b = affine.load %a[1, 1] : memref<?x2xf64>
    %b11b = affine.load %b[1, 1] : memref<?x2xf64>
    %p111 = arith.mulf %a11b, %b11b : f64
    %s111 = arith.subf %s110, %p111 : f64
    affine.store %s111, %c[1, 1] : memref<?x2xf64>
    return
  }
}

// CHECK-LABEL: func.func @arbitrary_matvec_name
// CHECK-NEXT: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%arg0, %arg1
// CHECK-SAME: outs(%arg2
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: linalg.yield
// CHECK: return

// CHECK-LABEL: func.func @arbitrary_matmul_name
// CHECK-NEXT: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%arg1, %arg0
// CHECK-SAME: outs(%arg2
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: linalg.yield
// CHECK: return
