#include <memory>
#include "mlir/Pass/Pass.h"
namespace mlir {
    namespace polygeist {
        std::unique_ptr<OperationPass<FuncOp>> createMem2RegPass();
        std::unique_ptr<OperationPass<FuncOp>> createLoopRestructurePass();
        std::unique_ptr<OperationPass<FuncOp>> replaceAffineCFGPass();
        std::unique_ptr<Pass> createCanonicalizeForPass();
        std::unique_ptr<Pass> createRaiseSCFToAffinePass();
        std::unique_ptr<OperationPass<FuncOp>> detectReductionPass();
        std::unique_ptr<OperationPass<FuncOp>> createParallelLowerPass();
    }
}

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace scf {
class SCFDialect;
} // end namespace scf

namespace memref {
class MemRefDialect;
} // end namespace memref

class AffineDialect;
class StandardOpsDialect;
namespace LLVM {
    class LLVMDialect;
}

#define GEN_PASS_CLASSES
#include "polygeist/Passes/Passes.h.inc"

} // end namespace mlir