// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed

// RUN: mkdir -p %t/schedules
// RUN: mkdir -p %t/accesses
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-import-schedules=%t/schedules $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed

// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s && find %t/schedules/ %t/accesses/ -type f -print0 | sort -z | xargs -0r cat | FileCheck --check-prefix=ISL_OUT %s

// ISL_OUT-NEXT: accesses:
// ISL_OUT-NEXT:   - S0:
// ISL_OUT-NEXT:       reads:
// ISL_OUT-NEXT:       writes:
// ISL_OUT-NEXT:         - "[P0] -> { A1[i0, i1] : i1 = 1 + i0 }"
// ISL_OUT-NEXT: accesses:
// ISL_OUT-NEXT:   - S1:
// ISL_OUT-NEXT:       reads:
// ISL_OUT-NEXT:       writes:
// ISL_OUT-NEXT:         - "[P0] -> { A1[i0, i1] :  }"
// ISL_OUT-NEXT: accesses:
// ISL_OUT-NEXT:   - S2:
// ISL_OUT-NEXT:       reads:
// ISL_OUT-NEXT:       writes:
// ISL_OUT-NEXT:         - "[P0] -> { A1[i0, i1] :  }"
// ISL_OUT-NEXT: accesses:
// ISL_OUT-NEXT:   - S3:
// ISL_OUT-NEXT:       reads:
// ISL_OUT-NEXT:       writes:
// ISL_OUT-NEXT:         - "[P0] -> { A1[i0, i1] :  }"
// ISL_OUT: { domain: "[P0] -> { S0[i0] : 0 <= i0 < P0 }", child: { schedule: "[P0] -> L0[{ S0[i0] -> [(i0)] }]" } }
// ISL_OUT-NEXT: { domain: "[P0] -> { S1[i0] : (i0) mod 2 = 0 and 0 <= i0 < P0 }", child: { schedule: "[P0] -> L0[{ S1[i0] -> [(i0)] }]" } }
// ISL_OUT-NEXT: { domain: "[P0] -> { S2[i0] : (i0) mod 3 = 0 and 0 <= i0 < P0 }", child: { schedule: "[P0] -> L0[{ S2[i0] -> [(i0)] }]" } }
// ISL_OUT-NEXT: { domain: "[P0] -> { S3[i0] : (i0) mod 4 = 0 and 0 <= i0 < P0 }", child: { schedule: "[P0] -> L0[{ S3[i0] -> [(i0)] }]" } }

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
module {
  func.func private @foo()
  func.func @f1(%beta: f32,
            %C: memref<?xf32>,
            %N: index) {
    affine.for %i = 0 to #map()[%N] step 1 {
      affine.store %beta, %C[%i + 1] : memref<?xf32>
    }
    func.call @foo() : () -> ()
    // TODO these should be the same scop
    affine.for %i = 0 to #map()[%N] step 2 {
      affine.store %beta, %C[%i + 2] : memref<?xf32>
    }
    affine.for %i = 0 to #map()[%N] step 3 {
      affine.store %beta, %C[%i + 3] : memref<?xf32>
    }
    return
  }
  func.func @f2(%beta: f32,
            %C: memref<?xf32>,
            %N: index) {
    affine.for %i = 0 to #map()[%N] step 4{
      affine.store %beta, %C[%i + 4] : memref<?xf32>
    }
    return
  }
}
