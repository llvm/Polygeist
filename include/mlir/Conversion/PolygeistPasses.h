#ifndef MLIR_CONVERSION_POLYGEISTPASSES_H_
#define MLIR_CONVERSION_POLYGEISTPASSES_H_

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
namespace polygeist {
std::unique_ptr<Pass> createConvertPolygeistToLLVMPass();
std::unique_ptr<Pass>
createConvertPolygeistToLLVMPass(const LowerToLLVMOptions &options,
                                 bool useCStyleMemRef, bool onlyGpuModules,
                                 std::string gpuTarget);
} // namespace polygeist
} // namespace mlir

#endif // POLYGEISTPASSES_H_
