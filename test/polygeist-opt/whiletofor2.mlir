// RUN: polygeist-opt -allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func.func @w2f(%ub : i32) -> (i32, f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %2:2 = scf.while (%arg10 = %c0_i32, %arg12 = %cst, %ac = %true) : (i32, f32, i1) -> (i32, f32) {
      %3 = arith.cmpi ult, %arg10, %ub : i32
      %a = arith.andi %3, %ac : i1
      scf.condition(%a) %arg10, %arg12 : i32, f32
    } do {
    ^bb0(%arg10: i32, %arg12: f32):
      %c = "test.something"() : () -> (i1)
      %3 = arith.addf %arg12, %cst1 : f32
      %p = arith.addi %arg10, %c1_i32 : i32
      scf.yield %p, %3, %c : i32, f32, i1
    }
    return %2#0, %2#1 : i32, f32
  }

  func.func @w2f_inner(%ub : i32) -> (i32, f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %2:2 = scf.while (%arg10 = %c0_i32, %arg12 = %cst, %ac = %true) : (i32, f32, i1) -> (i32, f32) {
      %3 = arith.cmpi ult, %arg10, %ub : i32
      %a = arith.andi %3, %ac : i1
      scf.condition(%a) %arg10, %arg12 : i32, f32
    } do {
    ^bb0(%arg10: i32, %arg12: f32):
      %c = "test.something"() : () -> (i1)
      %r:2 = scf.if %c -> (i32, f32) {
        %3 = arith.addf %arg12, %cst1 : f32
        %p = arith.addi %arg10, %c1_i32 : i32
        scf.yield %p, %3 : i32, f32
      } else {
        scf.yield %arg10, %arg12 : i32, f32
      }
      scf.yield %r#0, %r#1, %c : i32, f32, i1
    }
    return %2#0, %2#1 : i32, f32
  }

  func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: i8, %arg1: index, %arg2: i32, %arg3: i32, %arg4: i32) -> i32 {
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0:2 = scf.while (%arg5 = %c0_i32, %arg6 = %arg0, %arg7 = %true) : (i32, i8, i1) -> (i8, i32) {
      %1 = arith.cmpi slt, %arg5, %arg2 : i32
      %2 = arith.andi %1, %arg7 : i1
      scf.condition(%2) %arg6, %arg5 : i8, i32
    } do {
    ^bb0(%arg5: i8, %arg6: i32):
      %1 = arith.addi %arg6, %c1_i32 : i32
      %2 = arith.cmpi ne, %arg6, %arg4 : i32
      %3 = scf.if %2 -> (i32) {
        scf.yield %1 : i32
      } else {
        scf.yield %arg6 : i32
      }
      scf.yield %3, %c0_i8, %2 : i32, i8, i1
    }
    return %0#1 : i32
  }
}

// CHECK:   func.func @w2f(%[[arg0:.+]]: i32) -> (i32, f32) {
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[cst:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[cst_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[false:.+]] = arith.constant false
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = arith.cmpi ugt, %[[arg0]], %[[c0_i32]] : i32
// CHECK-NEXT:     %[[V1:.+]]:2 = scf.if %[[V0]] -> (i32, f32) {
// CHECK-NEXT:       %[[V2:.+]] = arith.index_cast %[[arg0]] : i32 to index
// CHECK-NEXT:       %[[V3:.+]]:3 = scf.for %[[arg1:.+]] = %[[c0]] to %[[V2]] step %[[c1:.+]] iter_args(%[[arg2:.+]] = %[[c0_i32]], %[[arg3:.+]] = %[[cst_0]], %[[arg4:.+]] = %[[true]]) -> (i32, f32, i1) {
// CHECK-NEXT:         %[[V4:.+]]:3 = scf.if %[[arg4]] -> (i32, f32, i1) {
// CHECK-NEXT:           %[[V5:.+]] = "test.something"() : () -> i1
// CHECK-NEXT:           %[[V6:.+]] = arith.addf %[[arg3]], %[[cst]] : f32
// CHECK-NEXT:           %[[V7:.+]] = arith.addi %[[arg2]], %[[c1_i32]] : i32
// CHECK-NEXT:           scf.yield %[[V7]], %[[V6]], %[[V5]] : i32, f32, i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %[[arg2]], %[[arg3]], %[[false]] : i32, f32, i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[V4]]#0, %[[V4]]#1, %[[V4]]#2 : i32, f32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[V3]]#0, %[[V3]]#1 : i32, f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[c0_i32]], %[[cst_0]] : i32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V1]]#0, %[[V1]]#1 : i32, f32
// CHECK-NEXT:   }

// CHECK:   func.func @w2f_inner(%[[arg0:.+]]: i32) -> (i32, f32) {
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[cst:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[cst_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[false:.+]] = arith.constant false
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = arith.cmpi ugt, %[[arg0]], %[[c0_i32]] : i32
// CHECK-NEXT:     %[[V1:.+]]:2 = scf.if %[[V0]] -> (i32, f32) {
// CHECK-NEXT:       %[[V2:.+]] = arith.index_cast %[[arg0]] : i32 to index
// CHECK-NEXT:       %[[V3:.+]]:3 = scf.for %[[arg1:.+]] = %[[c0]] to %[[V2]] step %[[c1:.+]] iter_args(%[[arg2:.+]] = %[[c0_i32]], %[[arg3:.+]] = %[[cst_0]], %[[arg4:.+]] = %[[true]]) -> (i32, f32, i1) {
// CHECK-NEXT:         %[[V4:.+]]:3 = scf.if %[[arg4]] -> (i32, f32, i1) {

// CHECK-NEXT:           %[[V5:.+]] = "test.something"() : () -> i1
// CHECK-NEXT:           %[[V6:.+]]:2 = scf.if %[[V5]] -> (i32, f32) {
// CHECK-NEXT:             %[[V7:.+]] = arith.addf %[[arg3]], %[[cst]] : f32
// CHECK-NEXT:             %[[V8:.+]] = arith.addi %[[arg2]], %[[c1_i32]] : i32
// CHECK-NEXT:             scf.yield %[[V8]], %[[V7]] : i32, f32
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %[[arg2]], %[[arg3]] : i32, f32
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[V6]]#0, %[[V6]]#1, %[[V5]] : i32, f32, i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %[[arg2]], %[[arg3]], %[[false]] : i32, f32, i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[V4]]#0, %[[V4]]#1, %[[V4]]#2 : i32, f32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[V3]]#0, %[[V3]]#1 : i32, f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[c0_i32]], %[[cst_0]] : i32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V1]]#0, %[[V1]]#1 : i32, f32
// CHECK-NEXT:   }

// CHECK:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%[[arg0:.+]]: i8, %[[arg1:.+]]: index, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32) -> i32 {
// CHECK-DAG:     %[[c0_i8:.+]] = arith.constant 0 : i8
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[false:.+]] = arith.constant false
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[V0:.+]] = arith.cmpi sgt, %[[arg2]], %[[c0_i32]] : i32
// CHECK-DAG:     %[[V1:.+]] = scf.if %[[V0]] -> (i32) {
// CHECK-DAG:       %[[V2:.+]] = arith.index_cast %[[arg2]] : i32 to index
// CHECK-NEXT:       %[[V3]]:3 = scf.for %[[arg5:.+]] = %[[c0]] to %[[V2]] step %[[c1:.+]] iter_args(%[[arg6:.+]] = %[[arg0]], %[[arg7:.+]] = %[[c0_i32]], %[[arg8:.+]] = %[[true]]) -> (i8, i32, i1) {
// CHECK-NEXT:         %[[V4:.+]]:3 = scf.if %[[arg8]] -> (i8, i32, i1) {
// CHECK-NEXT:           %[[V5:.+]] = arith.addi %[[arg7]], %[[c1_i32]] : i32
// CHECK-NEXT:           %[[V6:.+]] = arith.cmpi ne, %[[arg7]], %[[arg4]] : i32
// CHECK-NEXT:           %[[V7:.+]] = scf.if %[[V6]] -> (i32) {
// CHECK-NEXT:             scf.yield %[[V5]] : i32
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %[[arg7]] : i32
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[c0_i8]], %[[V7]], %[[V6]] : i8, i32, i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %[[arg6]], %[[arg7]], %[[false]] : i8, i32, i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[V4]]#0, %[[V4]]#1, %[[V4]]#2 : i8, i32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[V3]]#1 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[c0_i32]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V1]] : i32
// CHECK-NEXT:   }
