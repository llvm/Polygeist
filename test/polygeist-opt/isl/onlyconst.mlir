// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed

#map = affine_map<()[s0] -> (s0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?xf32>,
            %A: memref<?xf32>,
            %B: memref<?xf32>,
            %S: index,
            %N: index) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        affine.store %beta, %C[0] : memref<?xf32>
      }
    }
    return
  }
}
// RUN: mkdir -p %t/schedules
// RUN: mkdir -p %t/accesses
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s && find %t/schedules/ %t/accesses/ -type f -print0 | sort -z | xargs -0r cat | FileCheck --check-prefix=ISL_OUT %s
// ISL_OUT: domain: "{ S0[i0, i1] : 0 <= i0 <= 9 and 0 <= i1 <= 19 }"
// ISL_OUT: accesses:
// ISL_OUT:   - S0:
// ISL_OUT:       reads:
// ISL_OUT:       writes:
// ISL_OUT:         - "{ [i0, i1] -> A1[o0] : o0 = 0 }"
// ISL_OUT: { domain: "{ S0[i0, i1] : 0 <= i0 <= 9 and 0 <= i1 <= 19 }", child: { schedule: "L1[{ S0[i0, i1] -> [(i0)] }]", child: { schedule: "L0[{ S0[i0, i1] -> [(i1)] }]" } } }
