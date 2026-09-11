// RUN: polygeist-opt --raise-affine-to-linalg %s | FileCheck %s

module {
  func.func @hybrid_guarded_load(%in: memref<?xf32>, %out: memref<?xf32>,
                                 %n: index) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %c = 0 to 2 {
      affine.for %oh = 0 to 3 {
        affine.for %ow = 0 to 4 {
          %ok = arith.cmpi ult, %ow, %n : index
          %v = scf.if %ok -> (f32) {
            %idx0 = arith.muli %c, %n : index
            %idx1 = arith.addi %idx0, %ow : index
            %x = memref.load %in[%idx1] : memref<?xf32>
            scf.yield %x : f32
          } else {
            scf.yield %cst : f32
          }
          affine.store %v, %out[%ow + %oh * 4 + %c * 12] : memref<?xf32>
        }
      }
    }
    return
  }
}

// CHECK-DAG: #[[OUT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2 + d1 * 4 + d0 * 12)>
// CHECK-DAG: #[[ID_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @hybrid_guarded_load
// CHECK-NOT: affine.for
// CHECK: polygeist.submap
// CHECK-SAME: map = #[[OUT_MAP]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[ID_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: outs(
// CHECK: ^bb0(%{{.*}}: f32):
// CHECK: linalg.index 0
// CHECK: linalg.index 2
// CHECK: scf.if
// CHECK: memref.load
// CHECK: linalg.yield
// CHECK-NOT: affine.for
// CHECK: return
