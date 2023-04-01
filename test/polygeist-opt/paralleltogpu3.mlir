// RUN: polygeist-opt --lower-affine --canonicalize --convert-parallel-to-gpu1 --canonicalize %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func @f1() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1) = (0) to (10000000) {
        "polygeist.gpu_block"(%a1, %mc1, %mc1) ({
          affine.parallel (%a2) = (0) to (1024) {
            "polygeist.gpu_thread"(%a2, %mc1, %mc1) ({
              func.call @use(%a1) : (index) -> ()
              func.call @use(%a2) : (index) -> ()
              "polygeist.polygeist_yield"() : () -> ()
            }) : (index, index, index) -> ()
            affine.yield
          }
          "polygeist.polygeist_yield"() : () -> ()
        }) : (index, index, index) -> ()
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
  func.func @f2() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1) = (0) to (1) {
        "polygeist.gpu_block"(%mc1, %mc1, %mc1) ({
          affine.parallel (%a2) = (0) to (1024) {
            "polygeist.gpu_thread"(%a2, %mc1, %mc1) ({
              func.call @use(%a1) : (index) -> ()
              func.call @use(%a2) : (index) -> ()
              "polygeist.polygeist_yield"() : () -> ()
            }) : (index, index, index) -> ()
            affine.yield
          }
          "polygeist.polygeist_yield"() : () -> ()
        }) : (index, index, index) -> ()
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
  func.func @f3() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      "polygeist.gpu_block"(%mc1, %mc1, %mc1) ({
          affine.parallel (%a2) = (0) to (1024) {
          "polygeist.gpu_thread"(%a2, %mc1, %mc1) ({
              func.call @use(%a2) : (index) -> ()
              "polygeist.polygeist_yield"() : () -> ()
          }) : (index, index, index) -> ()
          }
          "polygeist.polygeist_yield"() : () -> ()
      }) : (index, index, index) -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
}
