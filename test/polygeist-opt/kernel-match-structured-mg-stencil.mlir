// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py --enable-structured-rewrite %s 2>&1 | sed '/^\/\/ CHECK/d' | FileCheck %s

#row = affine_map<(d0) -> (d0)>
#grid = affine_map<(d0)[s0, s1, s2] -> (s0, d0 + s2)>

module {
  // This is the raised three-stage shape of NPB MG resid.  The first two
  // generics build row temporaries; Egglog must fuse them with the epilogue
  // before the external cuDNN call can be emitted.
  func.func private @resid(
      %u: memref<?x?xf64>, %v: memref<?x?xf64>, %r: memref<?x?xf64>,
      %n1: i32, %n2: i32, %n3: i32, %a: memref<?xf64>, %level: i32) {
    %width = arith.index_cast %n1 : i32 to index
    %height = arith.index_cast %n2 : i32 to index
    %depth = arith.index_cast %n3 : i32 to index
    %t1 = memref.alloca() : memref<19xf64>
    %t2 = memref.alloca() : memref<19xf64>
    affine.for %z = 1 to %depth {
      affine.for %y = 1 to %height {
        %u0 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %u1 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %u2 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %u3 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %t1v = polygeist.submap(%t1, %width) {map = #row} :
            (memref<19xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#row, #row, #row, #row, #row],
                        iterator_types = ["parallel"]}
            ins(%u0, %u1, %u2, %u3 : memref<?xf64>, memref<?xf64>,
                memref<?xf64>, memref<?xf64>)
            outs(%t1v : memref<?xf64>) {
        ^bb0(%x0: f64, %x1: f64, %x2: f64, %x3: f64, %out: f64):
          %s0 = arith.addf %x0, %x1 : f64
          %s1 = arith.addf %s0, %x2 : f64
          %s2 = arith.addf %s1, %x3 : f64
          linalg.yield %s2 : f64
        }
        %q0 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %q1 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %q2 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %q3 = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %t2v = polygeist.submap(%t2, %width) {map = #row} :
            (memref<19xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#row, #row, #row, #row, #row],
                        iterator_types = ["parallel"]}
            ins(%q0, %q1, %q2, %q3 : memref<?xf64>, memref<?xf64>,
                memref<?xf64>, memref<?xf64>)
            outs(%t2v : memref<?xf64>) {
        ^bb0(%x0: f64, %x1: f64, %x2: f64, %x3: f64, %out: f64):
          %s0 = arith.addf %x0, %x1 : f64
          %s1 = arith.addf %s0, %x2 : f64
          %s2 = arith.addf %s1, %x3 : f64
          linalg.yield %s2 : f64
        }
        %vv = polygeist.submap(%v, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %a0 = polygeist.submap(%a, %width) {map = #row} :
            (memref<?xf64>, index) -> memref<?xf64>
        %uc = polygeist.submap(%u, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %a2 = polygeist.submap(%a, %width) {map = #row} :
            (memref<?xf64>, index) -> memref<?xf64>
        %a3 = polygeist.submap(%a, %width) {map = #row} :
            (memref<?xf64>, index) -> memref<?xf64>
        %rv = polygeist.submap(%r, %z, %y, %width) {map = #grid} :
            (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        linalg.generic {
            indexing_maps = [#row, #row, #row, #row, #row, #row, #row,
                             #row, #row, #row, #row],
            iterator_types = ["parallel"]}
            ins(%vv, %a0, %uc, %a2, %t2v, %t1v, %t1v, %a3, %t2v, %t2v :
                memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>,
                memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>,
                memref<?xf64>, memref<?xf64>)
            outs(%rv : memref<?xf64>) {
        ^bb0(%vin: f64, %wa0: f64, %center: f64, %wa2: f64,
             %e0: f64, %e1: f64, %e2: f64, %wa3: f64,
             %c0: f64, %c1: f64, %out: f64):
          %p0 = arith.mulf %wa0, %center : f64
          %r0 = arith.subf %vin, %p0 : f64
          %e01 = arith.addf %e0, %e1 : f64
          %es = arith.addf %e01, %e2 : f64
          %p1 = arith.mulf %wa2, %es : f64
          %r1 = arith.subf %r0, %p1 : f64
          %cs = arith.addf %c0, %c1 : f64
          %p2 = arith.mulf %wa3, %cs : f64
          %result = arith.subf %r1, %p2 : f64
          linalg.yield %result : f64
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func private @resid
// CHECK: %[[A0:.*]] = memref.load %a[%{{.*}}] : memref<?xf64>
// CHECK: %[[A2:.*]] = memref.load %a[%{{.*}}] : memref<?xf64>
// CHECK: %[[A3:.*]] = memref.load %a[%{{.*}}] : memref<?xf64>
// CHECK: kernel.launch @cudnnStencil3DSymmetric_f64_memref(%u, %v, %r,
// CHECK-NOT: affine.for
