// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed
#map = affine_map<()[s0] -> (s0)>
module {
  func.func @gemm(%alpha: f32, %beta: f32,
            %C: memref<?x?xf32>,
            %A: memref<?x?xf32>,
            %B: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %NI = memref.dim %C, %c0 : memref<?x?xf32>
    %NJ = memref.dim %C, %c1 : memref<?x?xf32>
    %NK = memref.dim %A, %c1 : memref<?x?xf32>

    affine.for %i = 0 to #map()[%NI] {
      affine.for %j = 0 to #map()[%NJ] {
        %0 = affine.load %C[%i, %j] : memref<?x?xf32>
        %1 = arith.mulf %0, %beta : f32
        affine.store %1, %C[%i, %j] : memref<?x?xf32>
      }

      affine.for %j = 0 to #map()[%NJ] {
        affine.for %k = 0 to #map()[%NK] {
          %2 = affine.load %A[%i, %k] : memref<?x?xf32>
          %3 = arith.mulf %alpha, %2 : f32
          %4 = affine.load %B[%k, %j] : memref<?x?xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = affine.load %C[%i, %j] : memref<?x?xf32>
          %7 = arith.addf %6, %5 : f32
          affine.store %7, %C[%i, %j] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
// RUN: mkdir -p %t/schedules
// RUN: mkdir -p %t/accesses
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s && find %t/schedules/ %t/accesses/ -type f -print0 | sort -z | xargs -0r cat | FileCheck --check-prefix=ISL_OUT %s
// ISL_OUT: domain: "[P0, P1, P2] -> { S0[i0, i1] : 0 <= i0 < P2 and 0 <= i1 < P0; S1[i0, i1, i2] : 0 <= i0 < P2 and 0 <= i1 < P0 and 0 <= i2 < P1 }"
// ISL_OUT: accesses:
// ISL_OUT:   - S0:
// ISL_OUT:       reads:
// ISL_OUT:         - "[P0, P1, P2] -> { [i0, i1] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
// ISL_OUT:       writes:
// ISL_OUT:         - "[P0, P1, P2] -> { [i0, i1] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
// ISL_OUT:   - S1:
// ISL_OUT:       reads:
// ISL_OUT:         - "[P0, P1, P2] -> { [i0, i1, i2] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
// ISL_OUT:         - "[P0, P1, P2] -> { [i0, i1, i2] -> A2[o0, o1] : o0 = i0 and o1 = i2 }"
// ISL_OUT:         - "[P0, P1, P2] -> { [i0, i1, i2] -> A3[o0, o1] : o0 = i2 and o1 = i1 }"
// ISL_OUT:       writes:
// ISL_OUT:         - "[P0, P1, P2] -> { [i0, i1, i2] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
// ISL_OUT: { domain: "[P0, P1, P2] -> { S0[i0, i1] : 0 <= i0 < P2 and 0 <= i1 < P0; S1[i0, i1, i2] : 0 <= i0 < P2 and 0 <= i1 < P0 and 0 <= i2 < P1 }", child: { schedule: "[P0, P1, P2] -> L3[{ S0[i0, i1] -> [(i0)]; S1[i0, i1, i2] -> [(i0)] }]", child: { sequence: [ { filter: "[P0, P1, P2] -> { S0[i0, i1] }", child: { schedule: "[P0, P1, P2] -> L0[{ S0[i0, i1] -> [(i1)] }]" } }, { filter: "[P0, P1, P2] -> { S1[i0, i1, i2] }", child: { schedule: "[P0, P1, P2] -> L2[{ S1[i0, i1, i2] -> [(i1)] }]", child: { schedule: "[P0, P1, P2] -> L1[{ S1[i0, i1, i2] -> [(i2)] }]" } } } ] } } }
