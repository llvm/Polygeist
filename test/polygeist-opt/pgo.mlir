// RUN: polygeist-opt --lower-alternatives --convert-polygeist-to-llvm --polygeist-alternatives-mode=pgo_prof %s | FileCheck %s

module {
  func.func private @wow0()
  func.func private @wow1()
  func.func private @wow2()
  func.func @f() {
    "polygeist.alternatives"() ({
      func.call @wow0() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      func.call @wow1() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["", ""], alternatives.type = "gpu_kernel"} : () -> ()

    return
  }
}

// CHECK-LABEL:   llvm.mlir.global internal constant @kernelId.0

// CHECK-LABEL:   llvm.func @f() {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @kernelId.0 : !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_3]][0, 0] : (!llvm.ptr) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.call @mgpurtPGOGetAlternative(%[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = llvm.icmp "eq" %[[VAL_5]], %[[VAL_1]] : i32
// CHECK:           llvm.cond_br %[[VAL_6]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           llvm.call @mgpurtPGOStart(%[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr, i32) -> ()
// CHECK:           llvm.call @wow0() : () -> ()
// CHECK:           llvm.call @mgpurtPGOEnd(%[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr, i32) -> ()
// CHECK:           llvm.br ^bb6
// CHECK:         ^bb2:
// CHECK:           %[[VAL_7:.*]] = llvm.icmp "eq" %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           llvm.cond_br %[[VAL_7]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           llvm.call @mgpurtPGOStart(%[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr, i32) -> ()
// CHECK:           llvm.call @wow1() : () -> ()
// CHECK:           llvm.call @mgpurtPGOEnd(%[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr, i32) -> ()
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb4:
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb5:
// CHECK:           llvm.br ^bb6
// CHECK:         ^bb6:
// CHECK:           llvm.return
