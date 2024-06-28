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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    affine.for %i = 0 to #map()[%N] {
      affine.if #set1(%i)[%S] {
        %4 = affine.load %B[%i] : memref<?xf32>
        affine.store %4, %C[%i] : memref<?xf32>
      } else {
        %4 = affine.load %A[%i] : memref<?xf32>
        affine.store %4, %C[%i] : memref<?xf32>
      }
    }
    return
  }
}
// RUN: mkdir -p %t/schedules
// RUN: mkdir -p %t/accesses
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s && find %t/schedules/ %t/accesses/ -type f -print0 | sort -z | xargs -0r cat | FileCheck --check-prefix=ISL_OUT %s
// ISL_OUT: domain: "[P0, P1] -> { S1[i0] : i0 >= P0 and 0 <= i0 < P1; S0[i0] : 0 <= i0 < P1 and i0 < P0 }"
// ISL_OUT: accesses:
// ISL_OUT:   - S0:
// ISL_OUT:       reads:
// ISL_OUT:         - "[P0, P1] -> { [i0] -> A1[o0] : o0 = i0 }"
// ISL_OUT:       writes:
// ISL_OUT:         - "[P0, P1] -> { [i0] -> A2[o0] : o0 = i0 }"
// ISL_OUT:   - S1:
// ISL_OUT:       reads:
// ISL_OUT:         - "[P0, P1] -> { [i0] -> A3[o0] : o0 = i0 }"
// ISL_OUT:       writes:
// ISL_OUT:         - "[P0, P1] -> { [i0] -> A2[o0] : o0 = i0 }"
// ISL_OUT: { domain: "[P0, P1] -> { S1[i0] : i0 >= P0 and 0 <= i0 < P1; S0[i0] : 0 <= i0 < P1 and i0 < P0 }", child: { schedule: "[P0, P1] -> L0[{ S1[i0] -> [(i0)]; S0[i0] -> [(i0)] }]", child: { sequence: [ { filter: "[P0, P1] -> { S0[i0] }" }, { filter: "[P0, P1] -> { S1[i0] }" } ] } } }
