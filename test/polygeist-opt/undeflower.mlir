// RUN: polygeist-opt --convert-polygeist-to-llvm %s | FileCheck %s

module {
  func.func @f() -> index {
    %a = "polygeist.undef"() : () -> index
  // CHECK: llvm.mlir.undef
    func.return %a : index
  }
}
