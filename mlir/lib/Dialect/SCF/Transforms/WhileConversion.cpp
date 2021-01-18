#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-while-to-for"

using namespace mlir;

namespace {
struct ConvertWhileToFor : public SCFWhileToSCFForBase<ConvertWhileToFor> {
  void runOnFunction() override;
};
} // namespace

void ConvertWhileToFor::runOnFunction() {
  // see: test/lib/Transforms/TestNumberOfExecutions.cp
  auto f = getFunction();
  f.dump();
  signalPassFailure();
}

std::unique_ptr<Pass> mlir::createWhileOpToForOpPass() {
  return std::make_unique<ConvertWhileToFor>();
}
