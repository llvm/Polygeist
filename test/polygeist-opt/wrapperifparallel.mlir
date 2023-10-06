// RUN: polygeist-opt --lower-affine --canonicalize-polygeist --convert-parallel-to-gpu1 --canonicalize-polygeist %s | FileCheck %s

// TODO we need versions that need gpu cache to split wrapper (from particlefilter), lud or sradv1 had an alloca in wrapper case

module {
  func.func private @use(%arg0: index)
  func.func private @wow()
  func.func @foo(%i : i1, %129 : index) -> () {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %err = "polygeist.gpu_wrapper"(%c512, %c1, %c1) ({
      scf.if %i {
        %c0_111 = arith.constant 0 : index
        %c1_112 = arith.constant 1 : index
        scf.parallel (%arg7) = (%c0_111) to (%129) step (%c1_112) {
          func.call @use(%arg7) : (index) -> ()
          scf.yield
        }
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                   %[[VAL_1:.*]]: index) {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           scf.if %[[VAL_0]] {
// CHECK:             "polygeist.alternatives"() ({
// CHECK:               %[[VAL_141:.*]] = arith.subi %[[VAL_1]], %[[VAL_7]] : index
// CHECK:               %[[VAL_142:.*]] = arith.divui %[[VAL_141]], %[[VAL_6]] : index
// CHECK:               %[[VAL_143:.*]] = arith.addi %[[VAL_142]], %[[VAL_7]] : index
// CHECK:               %[[VAL_144:.*]] = "polygeist.gpu_error"() ({
// CHECK:                 gpu.launch blocks(%[[VAL_145:.*]], %[[VAL_146:.*]], %[[VAL_147:.*]]) in (%[[VAL_148:.*]] = %[[VAL_143]], %[[VAL_149:.*]] = %[[VAL_7]], %[[VAL_150:.*]] = %[[VAL_7]]) threads(%[[VAL_151:.*]], %[[VAL_152:.*]], %[[VAL_153:.*]]) in (%[[VAL_154:.*]] = %[[VAL_6]], %[[VAL_155:.*]] = %[[VAL_7]], %[[VAL_156:.*]] = %[[VAL_7]]) {
// CHECK:                   %[[VAL_157:.*]] = gpu.block_id  x
// CHECK:                   %[[VAL_159:.*]] = gpu.thread_id  x
// CHECK:                   %[[VAL_160:.*]] = arith.muli %[[VAL_157]], %[[VAL_6]] : index
// CHECK:                   %[[VAL_161:.*]] = arith.addi %[[VAL_160]], %[[VAL_159]] : index
// CHECK:                   %[[VAL_162:.*]] = arith.cmpi ult, %[[VAL_161]], %[[VAL_1]] : index
// CHECK:                   scf.if %[[VAL_162]] {
// CHECK:                     func.call @use(%[[VAL_161]]) : (index) -> ()
// CHECK:                   }
// CHECK:                   gpu.terminator
// CHECK:                 }
// CHECK:                 "polygeist.polygeist_yield"() : () -> ()
// CHECK:               }) : () -> index
// CHECK:               "polygeist.polygeist_yield"() : () -> ()
// CHECK:             })
// CHECK:           }
// CHECK:           return
// CHECK:         }

