//===- AnnotateScop.cc --------------------------------------*- C++ -*-===//

#include "PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

namespace {
struct AnnotateScop : public polymer::AnnotateScopBase<AnnotateScop> {
  void runOnFunction() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    for (auto &name : includedFunctions)
      if (name == f.getName())
        return;

    f->setAttr("scop.ignored", b.getUnitAttr());
  } // namespace
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> polymer::createAnnotateScopPass() {
  return std::make_unique<AnnotateScop>();
}
