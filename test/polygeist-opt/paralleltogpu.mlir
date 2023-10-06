// RUN: polygeist-opt --lower-affine --canonicalize-polygeist --convert-parallel-to-gpu1 --canonicalize-polygeist %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func private @wow()
  func.func @f1() {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      func.call @wow() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           gpu.launch blocks(%[[VAL_1:.*]], %[[VAL_2:.*]], %[[VAL_3:.*]]) in (%[[VAL_4:.*]] = %[[VAL_0]], %[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_0]]) threads(%[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]]) in (%[[VAL_10:.*]] = %[[VAL_0]], %[[VAL_11:.*]] = %[[VAL_0]], %[[VAL_12:.*]] = %[[VAL_0]]) {
// CHECK:             func.call @wow() : () -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f2() {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1) = (0) to (10000000) {
        func.call @use(%a1) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f2() {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 9766 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 10000000 : index
// CHECK:           gpu.launch blocks(%[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]]) in (%[[VAL_7:.*]] = %[[VAL_2]], %[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_0]]) threads(%[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]]) in (%[[VAL_13:.*]] = %[[VAL_1]], %[[VAL_14:.*]] = %[[VAL_0]], %[[VAL_15:.*]] = %[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_18:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_19:.*]] = arith.muli %[[VAL_16]], %[[VAL_1]] : index
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:             %[[VAL_21:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_3]] : index
// CHECK:             scf.if %[[VAL_21]] {
// CHECK:               func.call @use(%[[VAL_20]]) : (index) -> ()
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f3() {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2) = (0, 0) to (16, 16) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f3() {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           gpu.launch blocks(%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) in (%[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_0]], %[[VAL_7:.*]] = %[[VAL_0]]) threads(%[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]]) in (%[[VAL_11:.*]] = %[[VAL_1]], %[[VAL_12:.*]] = %[[VAL_1]], %[[VAL_13:.*]] = %[[VAL_0]]) {
// CHECK:             %[[VAL_14:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_15:.*]] = gpu.thread_id  y
// CHECK:             func.call @use(%[[VAL_14]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_15]]) : (index) -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f4(%b1: index, %b2: index) {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 17, %b2, 32) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 17 : index
// CHECK:           gpu.launch blocks(%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) in (%[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_1]], %[[VAL_10:.*]] = %[[VAL_2]]) threads(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) in (%[[VAL_14:.*]] = %[[VAL_4]], %[[VAL_15:.*]] = %[[VAL_3]], %[[VAL_16:.*]] = %[[VAL_2]]) {
// CHECK:             %[[VAL_17:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_18:.*]] = gpu.block_id  y
// CHECK:             %[[VAL_19:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_20:.*]] = gpu.thread_id  y
// CHECK:             func.call @use(%[[VAL_17]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_19]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_18]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_20]]) : (index) -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f5(%b1: index, %b2: index) {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 5, %b2, 3) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// %a3 gets split into blockIdx.y and threadIdx.x, so use(%a3) becomes use(blockIdx.y x blockDim.x + threadIdx.x)
//
// CHECK-LABEL:   func.func @f5(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 16 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 32 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 5 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_161:.*]] = arith.subi %[[VAL_1]], %[[VAL_10]] : index
// CHECK:             %[[VAL_162:.*]] = arith.divui %[[VAL_161]], %[[VAL_7]] : index
// CHECK:             %[[VAL_163:.*]] = arith.addi %[[VAL_162]], %[[VAL_10]] : index
// CHECK:             %[[VAL_164:.*]] = "polygeist.gpu_error"() ({
// CHECK:               gpu.launch blocks(%[[VAL_165:.*]], %[[VAL_166:.*]], %[[VAL_167:.*]]) in (%[[VAL_168:.*]] = %[[VAL_0]], %[[VAL_169:.*]] = %[[VAL_163]], %[[VAL_170:.*]] = %[[VAL_10]]) threads(%[[VAL_171:.*]], %[[VAL_172:.*]], %[[VAL_173:.*]]) in (%[[VAL_174:.*]] = %[[VAL_7]], %[[VAL_175:.*]] = %[[VAL_9]], %[[VAL_176:.*]] = %[[VAL_8]]) {
// CHECK:                 %[[VAL_177:.*]] = gpu.block_id  x
// CHECK:                 %[[VAL_178:.*]] = gpu.block_id  y
// CHECK:                 %[[VAL_180:.*]] = gpu.thread_id  x
// CHECK:                 %[[VAL_181:.*]] = gpu.thread_id  y
// CHECK:                 %[[VAL_182:.*]] = gpu.thread_id  z
// CHECK:                 %[[VAL_183:.*]] = arith.muli %[[VAL_178]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_184:.*]] = arith.addi %[[VAL_183]], %[[VAL_180]] : index
// CHECK:                 %[[VAL_185:.*]] = arith.cmpi ult, %[[VAL_184]], %[[VAL_1]] : index
// CHECK:                 scf.if %[[VAL_185]] {
// CHECK:                   func.call @use(%[[VAL_177]]) : (index) -> ()
// CHECK:                   func.call @use(%[[VAL_181]]) : (index) -> ()
// CHECK:                   func.call @use(%[[VAL_184]]) : (index) -> ()
// CHECK:                   func.call @use(%[[VAL_182]]) : (index) -> ()
// CHECK:                 }
// CHECK:                 gpu.terminator
// CHECK:               }
// CHECK:               "polygeist.polygeist_yield"() : () -> ()
// CHECK:             }) : () -> index
// CHECK:           return
// CHECK:         }

  func.func @f6(%b1: index, %b2: index) {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 1023, %b2, 1025) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f6(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1025 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 1023 : index
// CHECK:           gpu.launch blocks(%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) in (%[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_1]], %[[VAL_10:.*]] = %[[VAL_3]]) threads(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) in (%[[VAL_14:.*]] = %[[VAL_4]], %[[VAL_15:.*]] = %[[VAL_2]], %[[VAL_16:.*]] = %[[VAL_2]]) {
// CHECK:             %[[VAL_17:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_18:.*]] = gpu.block_id  y
// CHECK:             %[[VAL_19:.*]] = gpu.block_id  z
// CHECK:             %[[VAL_20:.*]] = gpu.thread_id  x
// CHECK:             func.call @use(%[[VAL_17]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_20]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_18]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_19]]) : (index) -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
}
