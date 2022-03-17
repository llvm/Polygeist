//===- AnnotateScop.cc --------------------------------------*- C++ -*-===//

#include "PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

namespace {
struct AnnotateScop : public polymer::AnnotateScopBase<AnnotateScop> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    for (auto &name : includedFunctions)
      if (name == f.getName())
        return;

    f->setAttr("scop.ignored", b.getUnitAttr());
  } // namespace
};
} // namespace

std::unique_ptr<Pass> polymer::createAnnotateScopPass() {
  return std::make_unique<AnnotateScop>();
}
