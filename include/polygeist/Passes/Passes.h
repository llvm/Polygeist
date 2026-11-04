#ifndef POLYGEIST_DIALECT_POLYGEIST_PASSES_H
#define POLYGEIST_DIALECT_POLYGEIST_PASSES_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Dialect.h"
#include <memory>

#ifdef POLYGEIST_ENABLE_POLYMER
#include "polymer/Transforms/PlutoTransform.h"
#endif

enum PolygeistAlternativesMode { PAM_Static, PAM_PGO_Profile, PAM_PGO_Opt };
enum PolygeistGPUStructureMode {
  PGSM_Discard,
  PGSM_BlockThreadWrappers,
  PGSM_ThreadNoop,
  PGSM_BlockThreadNoops
};

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
namespace polygeist {
std::unique_ptr<Pass> createParallelLICMPass();
/// With `keepLoops`, LICM does not hoist a loop, or an op containing one, out
/// of an enclosing loop (cgeist --hls-keep-invariant-loops).
std::unique_ptr<Pass> createParallelLICMPass(bool keepLoops);
std::unique_ptr<Pass> createPolygeistMem2RegPass();
std::unique_ptr<Pass> createPolygeistAffineScalarReplacementPass();
std::unique_ptr<Pass> createLoopRestructurePass();
std::unique_ptr<Pass> createInnerSerializationPass();
std::unique_ptr<Pass> createSerializationPass();
std::unique_ptr<Pass> replaceAffineCFGPass();
std::unique_ptr<Pass> createOpenMPOptPass();
std::unique_ptr<Pass> createCanonicalizeForPass();
std::unique_ptr<Pass> createRaiseSCFToAffinePass();
std::unique_ptr<Pass> createCPUifyPass(StringRef method = "");
std::unique_ptr<Pass> createBarrierRemovalContinuation();
std::unique_ptr<Pass> detectReductionPass();
std::unique_ptr<Pass> createRemoveTrivialUsePass();
std::unique_ptr<Pass> createPolyhedralOptPass();
std::unique_ptr<Pass> createParallelLowerPass(
    bool wrapParallelOps = false,
    PolygeistGPUStructureMode gpuKernelStructureMode = PGSM_Discard);
std::unique_ptr<Pass> createConvertCudaRTtoCPUPass();
std::unique_ptr<Pass> createConvertCudaRTtoGPUPass();
std::unique_ptr<Pass> createConvertCudaRTtoHipRTPass();
std::unique_ptr<Pass> createFixGPUFuncPass();
std::unique_ptr<Pass> createSCFParallelLoopUnrollPass(int unrollFactor = 2);
std::unique_ptr<Pass>
createConvertPolygeistToLLVMPass(const LowerToLLVMOptions &options,
                                 bool useCStyleMemRef, bool onlyGpuModules,
                                 std::string gpuTarget);
std::unique_ptr<Pass> createConvertPolygeistToLLVMPass();
std::unique_ptr<Pass> createForBreakToWhilePass();
std::unique_ptr<Pass>
createConvertParallelToGPUPass1(std::string arch = "sm_60");
std::unique_ptr<Pass>
createConvertParallelToGPUPass2(bool emitGPUKernelLaunchBounds = true);
std::unique_ptr<Pass> createMergeGPUModulesPass();
std::unique_ptr<Pass> createConvertToOpaquePtrPass();
std::unique_ptr<Pass> createLowerAlternativesPass();
std::unique_ptr<Pass> createCollectKernelStatisticsPass();
std::unique_ptr<Pass> createPolygeistCanonicalizePass();
std::unique_ptr<Pass>
createPolygeistCanonicalizePass(const GreedyRewriteConfig &config,
                                ArrayRef<std::string> disabledPatterns,
                                ArrayRef<std::string> enabledPatterns);
std::unique_ptr<Pass> createGpuSerializeToCubinPass(
    StringRef arch, StringRef features, int llvmOptLevel, int ptxasOptLevel,
    std::string ptxasPath, std::string libDevicePath, bool outputIntermediate);
std::unique_ptr<Pass>
createGpuSerializeToHsacoPass(StringRef arch, StringRef features,
                              int llvmOptLevel, int hsaOptLevel,
                              std::string rocmPath, bool outputIntermediate);

void registerGpuSerializeToCubinPass();
void registerGpuSerializeToHsacoPass();

void populateForBreakToWhilePatterns(RewritePatternSet &patterns);

// Facts about the C program that the frontend records on ops as discardable
// attributes and the raising passes trust (AffineCFG, CanonicalizeFor). cgeist
// removes the polygeist.* ones from its output unless
// -keep-frontend-assumptions is given; hls.uncounted stays.

/// On an arith.addi / subi / muli: the C operation it computes has undefined
/// behaviour when its exact result does not fit the result type (signed
/// integer arithmetic at least as wide as int, C11 6.5p5, C++ [expr.pre]p4;
/// not under -fwrapv), so its result read signed is the exact result. On an
/// arith.index_cast from index to iN: the index value fits iN read signed.
constexpr llvm::StringLiteral kNoSignedWrapAttrName = "polygeist.nsw";
/// On an arith.index_cast from index to iN: the index value fits iN read
/// unsigned.
constexpr llvm::StringLiteral kNoUnsignedWrapAttrName = "polygeist.nuw";
/// On an scf.for that CanonicalizeFor raised from a while loop by assuming
/// that the loop terminates: its counter never wraps (see
/// -canonicalize-for-assume-termination in CanonicalizeFor.cpp). A BoolAttr:
/// true when the exit test reads the counter unsigned.
constexpr llvm::StringLiteral kCounterNoWrapAttrName =
    "polygeist.counter_nowrap";
/// On an scf.for / affine.for raised by that assumption whose exit test
/// compares its counter with a bound of a wider type that the counter is not
/// known to reach (`for (uint8_t i = 0; i < n; i++)`, int n): a loop whose
/// trip count the source does not bound. Kept in the output for hls-est
/// (Vitis does not collapse such a loop with its perfect inner nest,
/// hls-est-validate VITIS_NOTES.md N15).
constexpr llvm::StringLiteral kUncountedLoopAttrName = "hls.uncounted";
/// On a func.func: its body (or the body of a function inlined into it)
/// accesses a volatile-qualified object, which cgeist does not represent on
/// memref accesses. No loop in such a function, and no loop that calls it, is
/// assumed to terminate (C11 6.8.5p6 excludes loops that access volatile
/// objects; -canonicalize-for-assume-termination in CanonicalizeFor.cpp).
/// Recorded per function, not per loop: the frontend's loops become scf.while
/// only in LoopRestructure, and a mark on a loop or on an access op would not
/// survive the rewrites before CanonicalizeFor.
constexpr llvm::StringLiteral kVolatileAccessAttrName =
    "polygeist.volatile_access";

/// Canonicalization patterns that fold a chain of two marked add / sub with
/// constant operands, or of two marked mul by constants, into one marked op
/// (CanonicalizeFor.cpp). The arith patterns fold such chains through new ops
/// that would lose the mark.
void populateNoSignedWrapPatterns(RewritePatternSet &patterns);
/// Accesses through a memref.reinterpret_cast of an array argument whose
/// shape differs from the parameter's only in the first dimension use the
/// argument itself (PolygeistCanonicalize.cpp).
void populateArrayArgumentPatterns(RewritePatternSet &patterns);
} // namespace polygeist
} // namespace mlir

void fully2ComposeAffineMapAndOperands(
    mlir::PatternRewriter &rewriter, mlir::AffineMap *map,
    llvm::SmallVectorImpl<mlir::Value> *operands, mlir::DominanceInfo &DI);
bool isValidIndex(mlir::Value val);

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // end namespace arith

namespace omp {
class OpenMPDialect;
} // end namespace omp

namespace polygeist {
class PolygeistDialect;
} // end namespace polygeist

namespace scf {
class SCFDialect;
} // end namespace scf

namespace cf {
class ControlFlowDialect;
} // end namespace cf

namespace math {
class MathDialect;
} // end namespace math

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace func {
class FuncDialect;
}

namespace affine {
class AffineDialect;
}

namespace LLVM {
class LLVMDialect;
}

#define GEN_PASS_REGISTRATION
#include "polygeist/Passes/Passes.h.inc"

} // end namespace mlir

#endif // POLYGEIST_DIALECT_POLYGEIST_PASSES_H
