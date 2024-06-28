// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?xf32>,
            %A: memref<?x?xf32>,
            %B: memref<?xf32>,
            %S: index,
            %N: index) {
    affine.for %i = 0 to #map()[%N] {
      affine.store %beta, %C[%i] : memref<?xf32>
    }
    affine.for %i = 0 to #map()[%N] step 7 {
      affine.store %beta, %A[%i + 1, - 999 * %i + 666 * %N + 42] : memref<?x?xf32>
    }
    affine.for %i = 0 to #map()[%N] step 5 {
      affine.for %j = 0 to #map()[%N] {
        affine.store %beta, %B[%i + %j + 43] : memref<?xf32>
      }
    }
    return
  }
}
// RUN: mkdir -p %t/schedules
// RUN: mkdir -p %t/accesses
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s && find %t/schedules/ %t/accesses/ -type f -print0 | sort -z | xargs -0r cat | FileCheck --check-prefix=ISL_OUT %s
// ISL_OUT: accesses:
// ISL_OUT:   - S0:
// ISL_OUT:       reads:
// ISL_OUT:       writes:
// ISL_OUT:         - "[P0] -> { A1[i0, i1] : i1 = i0 }"
// ISL_OUT: { domain: "[P0] -> { S0[i0] : 0 <= i0 < P0 }", child: { schedule: "[P0] -> L0[{ S0[i0] -> [(i0)] }]" } }
