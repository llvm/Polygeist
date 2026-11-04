//===- HLSAutoInline.cc - Vitis automatic inlining decision ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See HLSAutoInline.h for the rule and its evidence.
//
//===----------------------------------------------------------------------===//

#include "HLSAutoInline.h"
#include "HLSPragma.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace cgeist::hls;

static llvm::cl::opt<bool> HLSAutoInline(
    "hls-auto-inline", llvm::cl::init(true),
    llvm::cl::desc("With --hls-annotate: keep the calls Vitis does not inline "
                   "automatically (a callee without an inline pragma whose "
                   "size x calls exceeds the threshold, a dataflow process); "
                   "=false inlines them as cgeist did before (for "
                   "comparisons)"));

static llvm::cl::opt<unsigned> HLSAutoInlineThreshold(
    "hls-auto-inline-threshold", llvm::cl::init(100),
    llvm::cl::desc("Largest size x calls Vitis inlines automatically (Vitis "
                   "2025.1: 100, VITIS_NOTES N18)"));

static llvm::cl::opt<bool> HLSAutoInlineDataflow(
    "hls-auto-inline-dataflow", llvm::cl::init(true),
    llvm::cl::desc("Keep the calls at the top level of a function with "
                   "'#pragma HLS dataflow' and the calls to such a function, "
                   "whatever their size (Vitis keeps dataflow processes, "
                   "VITIS_NOTES N18)"));

static llvm::cl::opt<bool> HLSAutoInlineRuntimeTripKept(
    "hls-auto-inline-runtime-trip-kept", llvm::cl::init(true),
    llvm::cl::desc("A loop marked for a complete unroll whose trip count is a "
                   "run-time value and whose maximum trip count exceeds "
                   "--hls-auto-inline-unroll-budget stays one loop in the "
                   "inline size count (VITIS_NOTES N161); off: it is "
                   "estimated at its maximum trip count"));

static llvm::cl::opt<bool> HLSAutoInlineCallSiteConstantTrips(
    "hls-auto-inline-callsite-constant-trips", llvm::cl::init(true),
    llvm::cl::desc("Before measuring a function with a loop to unroll "
                   "completely whose trip count is a run-time value, replace "
                   "each integer parameter that every call passes as the "
                   "same constant by that constant, so the loop is counted "
                   "at the trip count the calls give it (Vitis unrolls it "
                   "inside the callee then: VITIS_NOTES N161 (b), kernel "
                   "w9r21rev_r9b_rtunroll_big); off: such a loop counts "
                   "once (N161)"));

static llvm::cl::opt<unsigned> HLSAutoInlineUnrollBudget(
    "hls-auto-inline-unroll-budget", llvm::cl::init(65536),
    llvm::cl::desc("Designer knob: when the loops to unroll before measuring "
                   "would make a function larger than this many "
                   "instructions, its size and calls are estimated from the "
                   "trip counts instead of unrolling it (such a function is "
                   "kept by every caller; Vitis itself does not finish "
                   "unrolling e.g. lab2_hanchen)"));

static llvm::cl::opt<bool> HLSAutoInlineRotatedSize(
    "hls-auto-inline-rotated-size", llvm::cl::init(false),
    llvm::cl::desc("Measure a function's size in loop-rotated form as Vitis' "
                   "step 2 counts it (VITIS_NOTES N131): after the proxy, "
                   "loop-rotate, instcombine, simplifycfg, dce; lifetime "
                   "markers do not count; plus one per loop with a variable "
                   "trip count and one per constant-trip loop after the "
                   "first among the loops at the top of the function or "
                   "directly in a variable-trip loop (the guard / exit "
                   "blocks LLVM 7's rotation keeps); an llvm.fmuladd counts "
                   "as the fmul + fadd Vitis' front end emits"));

static llvm::cl::opt<bool> HLSAutoInlineCalleeUnroll(
    "hls-auto-inline-callee-unroll", llvm::cl::init(true),
    llvm::cl::desc("Before measuring, unroll the loops of every function "
                   "called (transitively) from a loop with a pipeline "
                   "pragma or from a pipelined function, as Vitis does "
                   "([HLS 214-291] 'implied by the pipeline pragma' for "
                   "callee loops, VITIS_NOTES N142: kernels w9r19p_iu_a / "
                   "iu_b / iu_d; an automatically pipelined loop does not, "
                   "iu_c); off: only the loops of the pipelined loop's own "
                   "function"));

static llvm::cl::opt<bool> HLSAutoInlineReport(
    "hls-auto-inline-report", llvm::cl::init(false),
    llvm::cl::desc("Print every automatic inlining decision (caller, callee, "
                   "size, calls) to stderr"));

static llvm::cl::opt<std::string> HLSAutoInlineDump(
    "hls-auto-inline-dump", llvm::cl::init(""), llvm::cl::Hidden,
    llvm::cl::desc("Debugging: print the LLVM IR the size of every function "
                   "whose name contains this text is measured on (after the "
                   "simplification) to stderr"));

namespace {

/// A loop's start location as clang writes it into the loop metadata (the
/// `for` / `while` / `do` keyword).
struct LoopLoc {
  std::string file; ///< basename of the presumed file name
  unsigned line = 0, col = 0;

  bool matches(const llvm::DebugLoc &start) const {
    return line == start.getLine() && col == start.getCol() &&
           file == llvm::sys::path::filename(start->getFilename());
  }
};

struct UnrollLoop : LoopLoc {
  int64_t factor = 0; ///< 0 = complete, -1 = `unroll off`
};

/// One function of clang's IR after the size recipe.
struct FunctionSize {
  int64_t size = 0;            ///< instructions without sext / zext / trunc
  bool returnsPointer = false; ///< pointer or reference return
  bool alwaysInline = false;   ///< `__attribute__((always_inline))`
  bool estimated = false;      ///< over the unroll budget: not unrolled
  /// Direct calls to functions with a body, per callee, after the unroll.
  std::map<std::string, int64_t> calls;
};

struct AutoInlineState {
  std::vector<UnrollLoop> unrollLoops;
  std::vector<LoopLoc> pipelineLoops;
  std::set<std::string> pipelineFunctions;
  /// The functions called, directly or through other functions, from a loop
  /// with a pipeline pragma or from a pipelined function (N142): their loops
  /// are unrolled before the size count like those of a pipelined function.
  std::set<std::string> reachedFromPipeline;
  /// The always_inline functions; measure() inlines them into their callers
  /// before it measures, while cgeist's MLIR still calls them.
  std::set<std::string> alwaysInline;
  std::map<std::string, FunctionSize> functions;
  bool measured = false;
};

AutoInlineState &state() {
  static AutoInlineState s;
  return s;
}

//===----------------------------------------------------------------------===//
// Measuring clang's IR
//===----------------------------------------------------------------------===//

/// The unroll factor of a loop: its own `#pragma HLS unroll`, else complete
/// when a pipelined loop encloses it or its function is pipelined (Vitis
/// unrolls those first: [HLS 214-291] "implied by the pipeline pragma", then
/// 214-186; w10rev rv_ai_pipe_inner, w10f_pipe_deep, w10f_fnpipe). nullopt:
/// not unrolled.
std::optional<int64_t> unrollFactor(llvm::Loop *L, bool functionPipelined) {
  const AutoInlineState &st = state();
  llvm::DebugLoc start = L->getStartLoc();
  if (start)
    for (const UnrollLoop &r : st.unrollLoops)
      if (r.matches(start))
        return r.factor < 0 ? std::nullopt : std::optional<int64_t>(r.factor);
  if (functionPipelined)
    return 0;
  for (llvm::Loop *p = L->getParentLoop(); p; p = p->getParentLoop())
    if (llvm::DebugLoc ps = p->getStartLoc())
      for (const LoopLoc &r : st.pipelineLoops)
        if (r.matches(ps))
          return 0;
  return std::nullopt;
}

/// Gives the loops Vitis unrolls before the inline decision (unrollFactor)
/// LLVM's forced-unroll metadata. Returns whether any loop was marked.
bool markUnrollLoops(llvm::Function &F, llvm::LoopInfo &LI) {
  const AutoInlineState &st = state();
  const bool functionPipelined =
      st.pipelineFunctions.count(F.getName().str()) > 0 ||
      st.reachedFromPipeline.count(F.getName().str()) > 0;
  if (st.unrollLoops.empty() && st.pipelineLoops.empty() && !functionPipelined)
    return false;
  bool any = false;
  llvm::LLVMContext &ctx = F.getContext();
  for (llvm::Loop *L : LI.getLoopsInPreorder()) {
    std::optional<int64_t> factor = unrollFactor(L, functionPipelined);
    if (!factor)
      continue;
    llvm::SmallVector<llvm::Metadata *, 4> ops{nullptr};
    if (llvm::MDNode *old = L->getLoopID())
      for (unsigned i = 1; i < old->getNumOperands(); ++i)
        ops.push_back(old->getOperand(i));
    if (*factor > 0)
      ops.push_back(llvm::MDNode::get(
          ctx, {llvm::MDString::get(ctx, "llvm.loop.unroll.count"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), *factor))}));
    else
      ops.push_back(llvm::MDNode::get(
          ctx, {llvm::MDString::get(ctx, "llvm.loop.unroll.full")}));
    llvm::MDNode *id = llvm::MDNode::getDistinct(ctx, ops);
    id->replaceOperandWith(0, id);
    L->setLoopID(id);
    any = true;
  }
  return any;
}

/// How many copies of `L`'s body its unroll metadata asks for: the trip
/// count (else the maximum one, when the budget holds it) for a complete
/// unroll, the factor for a partial one, 1 otherwise.
///
/// A loop whose trip count is a run-time value of its function (`for (i = 0;
/// i < length / 4; i++)` with `length` a parameter) is not unrolled before
/// the size count even when a pipeline pragma implies its complete unroll:
/// Vitis unrolls it only once inlining has made the trip constant
/// (VITIS_NOTES N161: class_ece8893 barry_walker lab2
/// `accumulate_quarter_HLS`, `cmplx 002` 63 instructions in 4 blocks, then
/// `[HLS 214-186] Unrolling loop ... in function 'softmax_HLS' completely with
/// a factor of 25` after its `[HLS 214-178]` inlining). Its maximum trip
/// count (2^29 there) is no unroll LLVM performs either.
uint64_t unrollCopies(llvm::Loop *L, llvm::ScalarEvolution &SE,
                      uint64_t budget) {
  llvm::MDNode *id = L->getLoopID();
  if (!id)
    return 1;
  if (llvm::GetUnrollMetadata(id, "llvm.loop.unroll.full")) {
    if (unsigned n = SE.getSmallConstantTripCount(L))
      return n;
    if (unsigned n = SE.getSmallConstantMaxTripCount(L))
      if (!HLSAutoInlineRuntimeTripKept || n <= budget)
        return n;
    return 1;
  }
  if (llvm::MDNode *count = llvm::GetUnrollMetadata(id, "llvm.loop.unroll.count"))
    if (auto *c = llvm::mdconst::extract_or_null<llvm::ConstantInt>(
            count->getOperand(1)))
      return std::max<uint64_t>(1, c->getZExtValue());
  return 1;
}

/// Whether `F` has a loop that unrollFactor() unrolls completely and whose
/// trip count is not a constant.
static bool hasRuntimeTripCompleteUnroll(llvm::Function &F, llvm::LoopInfo &LI,
                                         llvm::ScalarEvolution &SE) {
  const AutoInlineState &st = state();
  const bool functionPipelined =
      st.pipelineFunctions.count(F.getName().str()) > 0 ||
      st.reachedFromPipeline.count(F.getName().str()) > 0;
  for (llvm::Loop *L : LI.getLoopsInPreorder()) {
    std::optional<int64_t> factor = unrollFactor(L, functionPipelined);
    if (factor && *factor == 0 && !SE.getSmallConstantTripCount(L))
      return true;
  }
  return false;
}

/// Replaces each integer parameter of `F` that every call of `F` passes as
/// one and the same constant by that constant (interprocedural constant
/// propagation of the arguments). Returns whether anything was replaced.
/// Nothing happens when `F` has no call, or a use that is not the callee of
/// a direct call (an address taken).
///
/// VITIS_NOTES N161 (b): Vitis propagates such a constant into the callee
/// before it unrolls and measures it: kernel w9r21rev_r9b_rtunroll_big
/// (`accq(row, wl, d, 64)`, the one call; `for (i = 0; i < length / 4; i++)`
/// under `#pragma HLS unroll`): `[HLS 214-186] Unrolling loop
/// 'VITIS_LOOP_5_1' ... in function 'accq<short>' completely with a factor
/// of 16`, csynth_design_size.rpt accq<short> 895 after the unroll, kept
/// (own module grp_accq_short_s). Calls that pass different constants
/// (barry_walker lab2: `accumulate_quarter_HLS(..., dk)` and `(..., N)`) or
/// a run-time value (w9r21rev_r9a_rtunroll) leave the parameter alone, and
/// Vitis inlines those callees before it unrolls the loop (N161).
static bool propagateCallSiteConstants(llvm::Function &F) {
  llvm::SmallVector<llvm::CallBase *, 4> calls;
  for (llvm::Use &use : F.uses()) {
    auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
    if (!call || !call->isCallee(&use) ||
        call->arg_size() != F.arg_size())
      return false;
    calls.push_back(call);
  }
  if (calls.empty())
    return false;
  bool changed = false;
  for (llvm::Argument &arg : F.args()) {
    if (!arg.getType()->isIntegerTy() || arg.use_empty())
      continue;
    auto *value = llvm::dyn_cast<llvm::ConstantInt>(
        calls.front()->getArgOperand(arg.getArgNo()));
    if (!value)
      continue;
    if (!llvm::all_of(calls, [&](llvm::CallBase *call) {
          return call->getArgOperand(arg.getArgNo()) == value;
        }))
      continue;
    arg.replaceAllUsesWith(value);
    changed = true;
  }
  return changed;
}

/// Whether a call of `callee` counts as a call of another design function.
/// A declaration counts too: with several translation units, the callee may
/// be defined in another one (Vitis links the units before it inlines, so
/// `top` in a.cpp calling `big` of b.cpp follows the size rule; before, such
/// a call was not counted and the inliner always inlined it: class_ece8893
/// FLAT-acceleration's computeLogit / computeAttention, which Vitis keeps).
/// A callee no unit defines has no measured size and is left to the inliner
/// (MarkAutoInlinePass skips it); intrinsics never count.
static bool countsAsDesignCall(const llvm::Function *callee) {
  return callee && !callee->isIntrinsic();
}

/// `F`'s size and calls as if its marked loops were unrolled: every block
/// counted once per copy of each enclosing marked loop (no simplification).
/// Counting stops (saturates) above `budget`.
FunctionSize estimateUnrolled(llvm::Function &F, llvm::LoopInfo &LI,
                              llvm::ScalarEvolution &SE, uint64_t budget) {
  FunctionSize fs;
  std::map<llvm::Loop *, uint64_t> copies;
  auto copiesOf = [&](llvm::BasicBlock &BB) {
    uint64_t m = 1;
    for (llvm::Loop *L = LI.getLoopFor(&BB); L; L = L->getParentLoop()) {
      auto it = copies.find(L);
      if (it == copies.end())
        it = copies.emplace(L, unrollCopies(L, SE, budget)).first;
      m = std::min<uint64_t>(m * it->second, budget + 1);
    }
    return m;
  };
  uint64_t size = 0;
  for (llvm::BasicBlock &BB : F) {
    uint64_t m = copiesOf(BB);
    for (llvm::Instruction &I : BB) {
      if (llvm::isa<llvm::DbgInfoIntrinsic, llvm::SExtInst, llvm::ZExtInst,
                    llvm::TruncInst>(I))
        continue;
      size = std::min<uint64_t>(size + m, budget + 1);
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&I))
        if (llvm::Function *callee = call->getCalledFunction())
          if (countsAsDesignCall(callee))
            fs.calls[callee->getName().str()] += m;
    }
  }
  fs.size = static_cast<int64_t>(size);
  return fs;
}

/// The integer a pointer operand of a Xilinx builtin points at: the
/// allocated / element / global type, or the type of a store into it just
/// before `at` or of a load from it, descending into the first member of a
/// struct or array (an ap_int's `V`). Null when no integer is found.
static llvm::IntegerType *builtinOperandType(llvm::Value *ptr,
                                             llvm::Instruction *at) {
  llvm::Value *p = ptr->stripPointerCasts();
  llvm::Type *t = nullptr;
  if (auto *a = llvm::dyn_cast<llvm::AllocaInst>(p))
    t = a->getAllocatedType();
  else if (auto *g = llvm::dyn_cast<llvm::GEPOperator>(p))
    t = g->getResultElementType();
  else if (auto *gv = llvm::dyn_cast<llvm::GlobalVariable>(p))
    t = gv->getValueType();
  if (!t)
    for (llvm::Instruction *i = at->getPrevNode(); i && !t;
         i = i->getPrevNode())
      if (auto *st = llvm::dyn_cast<llvm::StoreInst>(i))
        if (st->getPointerOperand() == ptr)
          t = st->getValueOperand()->getType();
  if (!t)
    for (llvm::User *u : ptr->users())
      if (auto *ld = llvm::dyn_cast<llvm::LoadInst>(u)) {
        t = ld->getType();
        break;
      }
  while (t) {
    if (auto *it = llvm::dyn_cast<llvm::IntegerType>(t))
      return it;
    if (auto *st = llvm::dyn_cast<llvm::StructType>(t))
      t = st->getNumElements() ? st->getElementType(0) : nullptr;
    else if (auto *at2 = llvm::dyn_cast<llvm::ArrayType>(t))
      t = at2->getNumElements() ? at2->getElementType() : nullptr;
    else
      t = nullptr;
  }
  return nullptr;
}

/// cgeist lowers the Xilinx HLS builtins of the real Vitis headers
/// (include/xlx/xlx_prelude.h: __builtin_bit_*, __fpga_fifo_pop / push) in
/// its MLIR codegen; the LLVM IR measured here sees them as calls of external
/// functions that take their operands through pointers, which keeps every
/// operand in a stack slot (SROA cannot promote an escaping alloca): an
/// alloca, a store and the call per use, where Vitis's front end emits a
/// load, one `llvm.fpga.*` intrinsic call (or, for a concatenation, a shl
/// and an or) and a store (hls-llvm-project CGFPGABuiltin.cpp,
/// EmitBuiltinBit*), which its simplification turns into one instruction.
/// GradCAM's `cam_core` measured 102 with the external calls (kept) and is
/// inlined by Vitis ([HLS 214-178]). Rewrite each such call the way Vitis's
/// front end emits it, with an opaque `readnone` function standing for the
/// intrinsic, so that the operands' slots promote and the count matches.
/// Calls whose operand types cannot be found are left as they are.
static void lowerXilinxBuiltinsForMeasure(llvm::Module &M) {
  llvm::LLVMContext &ctx = M.getContext();
  auto opaque = [&](llvm::StringRef name, llvm::Type *ret,
                    llvm::ArrayRef<llvm::Type *> args, bool pure) {
    std::string full = ("__xlx_measure." + name).str();
    for (llvm::Type *a : args)
      full += "." + std::to_string(a->isIntegerTy()
                                       ? a->getIntegerBitWidth()
                                       : 0u);
    if (ret->isIntegerTy())
      full += ".r" + std::to_string(ret->getIntegerBitWidth());
    llvm::FunctionCallee fc = M.getOrInsertFunction(
        full, llvm::FunctionType::get(ret, args, false));
    if (auto *f = llvm::dyn_cast<llvm::Function>(fc.getCallee())) {
      f->addFnAttr(llvm::Attribute::NoUnwind);
      if (pure) {
        f->setDoesNotAccessMemory();
        f->addFnAttr(llvm::Attribute::WillReturn);
      }
    }
    return fc;
  };
  std::vector<llvm::CallInst *> calls;
  for (llvm::Function &F : M)
    for (llvm::Instruction &I : llvm::instructions(F))
      if (auto *c = llvm::dyn_cast<llvm::CallInst>(&I))
        if (llvm::Function *callee = c->getCalledFunction())
          if (callee->isDeclaration() &&
              (callee->getName().starts_with("__builtin_bit_") ||
               callee->getName() == "__fpga_fifo_pop" ||
               callee->getName() == "__fpga_fifo_push"))
            calls.push_back(c);
  for (llvm::CallInst *c : calls) {
    llvm::StringRef name = c->getCalledFunction()->getName();
    llvm::IRBuilder<> b(c);
    unsigned n = c->arg_size();
    auto ty = [&](unsigned i) { return builtinOperandType(c->getArgOperand(i), c); };
    auto load = [&](unsigned i, llvm::IntegerType *t) {
      return b.CreateLoad(t, c->getArgOperand(i));
    };
    llvm::Value *result = nullptr; // the call's own result, if any
    bool done = false;
    if ((name == "__builtin_bit_select" && n == 2) ||
        (name.ends_with("_reduce") && n == 1)) {
      if (llvm::IntegerType *t = ty(0)) {
        llvm::SmallVector<llvm::Value *, 2> args{load(0, t)};
        llvm::SmallVector<llvm::Type *, 2> types{t};
        if (n == 2) {
          args.push_back(c->getArgOperand(1));
          types.push_back(c->getArgOperand(1)->getType());
        }
        result = b.CreateCall(
            opaque(name.drop_front(10), c->getType(), types, true), args);
        done = true;
      }
    } else if (name == "__builtin_bit_concat" && n == 3) {
      llvm::IntegerType *r = ty(0), *hi = ty(1), *lo = ty(2);
      if (r && hi && lo) {
        llvm::Value *h = b.CreateZExtOrTrunc(load(1, hi), r);
        llvm::Value *l = b.CreateZExtOrTrunc(load(2, lo), r);
        llvm::Value *v = b.CreateOr(
            b.CreateShl(h, llvm::ConstantInt::get(r, lo->getBitWidth())), l);
        b.CreateStore(v, c->getArgOperand(0));
        done = true;
      }
    } else if (name.starts_with("__builtin_bit_") && n >= 2 &&
               c->getType()->isVoidTy()) {
      // set / part_select / part_set / from_string: result through arg 0,
      // the other pointer operands loaded, the rest passed on.
      if (llvm::IntegerType *r = ty(0)) {
        llvm::SmallVector<llvm::Value *, 4> args;
        llvm::SmallVector<llvm::Type *, 4> types;
        bool ok = true;
        for (unsigned i = 1; i < n && ok; ++i) {
          llvm::Value *a = c->getArgOperand(i);
          if (a->getType()->isPointerTy() &&
              name != "__builtin_bit_from_string") {
            llvm::IntegerType *t = ty(i);
            ok = t != nullptr;
            if (ok)
              a = load(i, t);
          }
          if (ok) {
            args.push_back(a);
            types.push_back(a->getType());
          }
        }
        if (ok) {
          llvm::Value *v =
              b.CreateCall(opaque(name.drop_front(10), r, types, true), args);
          b.CreateStore(v, c->getArgOperand(0));
          done = true;
        }
      }
    } else if (name == "__fpga_fifo_pop" && n == 2) {
      if (llvm::IntegerType *t = ty(1)) {
        llvm::Value *v = b.CreateCall(
            opaque("fifo_pop", t, {c->getArgOperand(0)->getType()}, false),
            {c->getArgOperand(0)});
        b.CreateStore(v, c->getArgOperand(1));
        done = true;
      }
    } else if (name == "__fpga_fifo_push" && n == 2) {
      if (llvm::IntegerType *t = ty(1)) {
        b.CreateCall(opaque("fifo_push", llvm::Type::getVoidTy(ctx),
                            {c->getArgOperand(0)->getType(), t}, false),
                     {c->getArgOperand(0), load(1, t)});
        done = true;
      }
    }
    if (!done)
      continue;
    if (result)
      c->replaceAllUsesWith(b.CreateZExtOrTrunc(result, c->getType()));
    else if (!c->use_empty())
      c->replaceAllUsesWith(llvm::UndefValue::get(c->getType()));
    c->eraseFromParent();
  }
}

/// Whether `F` is a member of one of the ap_int / ap_fixed classes or their
/// bases and proxies (ap_int_base, ap_fixed_base, ap_bit_ref, af_range_ref,
/// ...): its mangled name is a nested name whose first component is such a
/// class (`_ZN13ap_fixed_baseILi16E...EC2Ed`, `_ZNK11ap_int_base...`). The
/// headers mark these AP_NODEBUG, so they carry no debug location to go by;
/// the demangled name would not do either (a free template function's
/// starts with its return type, `ap_fixed<16, 2> hls::sinpi<...>`).
static bool isApClassHeaderFunction(const llvm::Function &F) {
  llvm::StringRef n = F.getName();
  if (!n.consume_front("_ZN"))
    return false;
  n.consume_front("K");
  unsigned len = 0;
  if (n.consumeInteger(10, len) || len < 3 || n.size() < len)
    return false;
  llvm::StringRef cls = n.take_front(len);
  return cls.starts_with("ap_") || cls.starts_with("af_");
}

/// Folds the stand-ins lowerXilinxBuiltinsForMeasure made whose operands
/// all became constants (after SROA and constant propagation), as LLVM's
/// constant folding of the `llvm.fpga.*` intrinsics does in Vitis's flow
/// (hls-llvm-project ConstantFolding.cpp): the double constructor of
/// ap_fixed_base selects the fields of a constant double's bits and folds
/// away. Returns whether anything was folded.
static bool foldConstantBuiltinStandIns(llvm::Function &F) {
  std::vector<std::pair<llvm::CallInst *, llvm::APInt>> folds;
  for (llvm::Instruction &I : llvm::instructions(F)) {
    auto *c = llvm::dyn_cast<llvm::CallInst>(&I);
    llvm::Function *callee = c ? c->getCalledFunction() : nullptr;
    if (!callee || !callee->getName().starts_with("__xlx_measure.") ||
        !c->getType()->isIntegerTy())
      continue;
    llvm::SmallVector<llvm::APInt, 4> a;
    for (llvm::Value *v : c->args()) {
      auto *ci = llvm::dyn_cast<llvm::ConstantInt>(v);
      if (!ci)
        break;
      a.push_back(ci->getValue());
    }
    if (a.size() != c->arg_size())
      continue;
    llvm::StringRef kind =
        callee->getName().drop_front(strlen("__xlx_measure.")).split('.').first;
    unsigned rw = c->getType()->getIntegerBitWidth();
    auto pos = [](const llvm::APInt &v) { return v.getZExtValue(); };
    std::optional<llvm::APInt> r;
    if (kind == "bit_select" && a.size() == 2 && pos(a[1]) < a[0].getBitWidth()) {
      r = llvm::APInt(rw, a[0][pos(a[1])] ? 1 : 0);
    } else if (kind == "bit_part_select" && a.size() == 3) {
      uint64_t lo = pos(a[1]), hi = pos(a[2]), w = a[0].getBitWidth();
      if (lo < w && hi < w) {
        llvm::APInt out(rw, 0);
        if (lo <= hi) {
          for (uint64_t k = 0; k <= hi - lo && k < rw; ++k)
            out.setBitVal(k, a[0][lo + k]);
        } else { // reversed range
          for (uint64_t k = 0; k <= lo - hi && k < rw; ++k)
            out.setBitVal(k, a[0][lo - k]);
        }
        r = out;
      }
    } else if (kind == "bit_set" && a.size() == 3 &&
               pos(a[2]) < a[0].getBitWidth()) {
      llvm::APInt out = a[0].zextOrTrunc(rw);
      if (pos(a[2]) < rw)
        out.setBitVal(pos(a[2]), a[1][0]);
      r = out;
    } else if (kind == "bit_part_set" && a.size() == 4) {
      uint64_t lo = pos(a[2]), hi = pos(a[3]);
      llvm::APInt out = a[0].zextOrTrunc(rw);
      if (lo <= hi && hi < rw) {
        for (uint64_t k = 0; k <= hi - lo; ++k)
          out.setBitVal(lo + k, k < a[1].getBitWidth() && a[1][k]);
        r = out;
      }
    } else if (kind.ends_with("_reduce") && a.size() == 1) {
      bool v = false;
      llvm::StringRef op = kind.drop_front(strlen("bit_"));
      if (op == "and_reduce" || op == "nand_reduce")
        v = a[0].isAllOnes();
      else if (op == "or_reduce" || op == "nor_reduce")
        v = !a[0].isZero();
      else if (op == "xor_reduce" || op == "nxor_reduce")
        v = a[0].popcount() % 2;
      else
        continue;
      if (op.starts_with("n"))
        v = !v;
      r = llvm::APInt(rw, v ? 1 : 0);
    }
    if (r)
      folds.push_back({c, *r});
  }
  for (auto &[c, v] : folds) {
    c->replaceAllUsesWith(llvm::ConstantInt::get(c->getType(), v));
    c->eraseFromParent();
  }
  return !folds.empty();
}

/// Whether a loop with a pipeline pragma encloses `BB` (the innermost loop
/// itself or any loop around it).
static bool inPipelinedLoop(llvm::BasicBlock &BB, llvm::LoopInfo &LI) {
  for (llvm::Loop *L = LI.getLoopFor(&BB); L; L = L->getParentLoop())
    if (llvm::DebugLoc start = L->getStartLoc())
      for (const LoopLoc &r : state().pipelineLoops)
        if (r.matches(start))
          return true;
  return false;
}

/// Fills AutoInlineState::reachedFromPipeline (N142): Vitis applies the
/// pipeline pragma's implied complete unroll to the loops of every function
/// the pipelined loop (or pipelined function) calls, transitively, before it
/// counts sizes for the automatic inlining ([HLS 214-291] for the callee
/// loop; kernel w9r19p_iu_a: callee size 17 -> 96, kept; iu_d: through an
/// intermediate function; hlsyn aes `aes_subBytes_1` 17 -> 96). One body
/// serves every caller, so a function reached from one pipelined loop is
/// measured unrolled for all of them. Runs on the module after the
/// always_inline bodies are inlined (their calls are the caller's).
static void computeReachedFromPipeline(llvm::Module &M,
                                       llvm::FunctionAnalysisManager &FAM) {
  AutoInlineState &st = state();
  st.reachedFromPipeline.clear();
  if (!HLSAutoInlineCalleeUnroll ||
      (st.pipelineLoops.empty() && st.pipelineFunctions.empty()))
    return;
  std::vector<llvm::Function *> work;
  auto reach = [&](llvm::Function *callee) {
    if (callee && !callee->isDeclaration() && !callee->isIntrinsic() &&
        st.reachedFromPipeline.insert(callee->getName().str()).second)
      work.push_back(callee);
  };
  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;
    const bool pipelined = st.pipelineFunctions.count(F.getName().str()) > 0;
    llvm::LoopInfo *LI = nullptr;
    if (!pipelined && !st.pipelineLoops.empty())
      LI = &FAM.getResult<llvm::LoopAnalysis>(F);
    for (llvm::BasicBlock &BB : F) {
      if (!pipelined && (!LI || !inPipelinedLoop(BB, *LI)))
        continue;
      for (llvm::Instruction &I : BB)
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&I))
          reach(call->getCalledFunction());
    }
    // Nothing cached from here may outlive the IR changes measure() makes
    // outside a pass manager (foldConstantBuiltinStandIns).
    FAM.invalidate(F, llvm::PreservedAnalyses::none());
  }
  while (!work.empty()) {
    llvm::Function *F = work.back();
    work.pop_back();
    for (llvm::Instruction &I : llvm::instructions(*F))
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&I))
        reach(call->getCalledFunction());
  }
  if (HLSAutoInlineReport)
    for (const std::string &name : st.reachedFromPipeline)
      llvm::errs() << "hls-auto-inline: '" << name
                   << "' is called from a pipelined loop or function: its "
                      "loops are unrolled before measuring\n";
}

/// N18's size recipe on every function of `M`: the always_inline callees
/// inlined (clang -O0 runs the always-inliner, so N18's proxy measures every
/// function with them inlined); sroa; the pragma loops unrolled; early-cse,
/// instcombine, simplifycfg, dce; then the instructions without the integer
/// casts, and the direct calls per callee.
void measure(llvm::Module &M) {
  llvm::PassBuilder PB;
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  auto parse = [&](llvm::StringRef text, llvm::FunctionPassManager &FPM) {
    if (llvm::Error err = PB.parsePassPipeline(FPM, text)) {
      llvm::errs() << "error: --hls-auto-inline: " << llvm::toString(std::move(err))
                   << "\n";
      return false;
    }
    return true;
  };
  // Vitis inlines every member of the ap_int / ap_fixed classes before it
  // measures: none of them is ever a row of a design-size report
  // (csynth_design_size.rpt, all td=off ground-truth runs), and the one the
  // Vitis headers do not mark AP_INLINE (always_inline), the
  // `ap_fixed_base(double)` constructor (etc/ap_fixed_base.h: "they're all
  // AP_INLINE expect ap_fixed_base(double d)"), has no row and no [HLS
  // 214-178] message either (GradCAM avg_pool: `const fm_t factor =
  // (fm_t)(16.0 / 49)` in avg_pool_core, which Vitis inlines; measured with
  // the constructor as a callee of its own, 106 and kept).
  for (llvm::Function &F : M)
    if (!F.isDeclaration() &&
        !F.hasFnAttribute(llvm::Attribute::AlwaysInline) &&
        isApClassHeaderFunction(F)) {
      F.removeFnAttr(llvm::Attribute::NoInline);
      F.removeFnAttr(llvm::Attribute::OptimizeNone);
      F.addFnAttr(llvm::Attribute::AlwaysInline);
    }
  for (llvm::Function &F : M)
    if (!F.isDeclaration() && F.hasFnAttribute(llvm::Attribute::AlwaysInline))
      state().alwaysInline.insert(F.getName().str());
  // As clang's -O0 pipeline: no lifetime markers for the inlined allocas.
  llvm::ModulePassManager alwaysInline;
  alwaysInline.addPass(
      llvm::AlwaysInlinerPass(/*InsertLifetimeIntrinsics=*/false));
  alwaysInline.run(M, MAM);
  lowerXilinxBuiltinsForMeasure(M);
  computeReachedFromPipeline(M, FAM);

  llvm::FunctionPassManager pre, post, rotated;
  if (!parse("sroa", pre) ||
      !parse("early-cse,instcombine<no-verify-fixpoint>,simplifycfg,dce", post) ||
      (HLSAutoInlineRotatedSize &&
       !parse("loop(loop-rotate),instcombine<no-verify-fixpoint>,simplifycfg,"
              "dce",
              rotated)))
    return;

  // A complete unroll of large nests can take LLVM practically forever
  // (lab2_hanchen: a 4 x 100 x 128 nest under a pipelined loop; Vitis's own
  // run of it ends in its unroll step without a result). Such a function is
  // far over the threshold whatever simplification follows, so its size and
  // calls are estimated from the trip counts instead.
  const uint64_t budget = HLSAutoInlineUnrollBudget;
  std::map<std::string, FunctionSize> estimates;
  // The callers' arguments must be SSA values (sroa) before a callee looks
  // at them (propagateCallSiteConstants); sroa is function-local and the
  // second run in the loop below finds nothing more to do.
  if (HLSAutoInlineCallSiteConstantTrips)
    for (llvm::Function &F : M)
      if (!F.isDeclaration())
        pre.run(F, FAM);
  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;
    pre.run(F, FAM);
    if (HLSAutoInlineCallSiteConstantTrips) {
      bool runtimeTrip = hasRuntimeTripCompleteUnroll(
          F, FAM.getResult<llvm::LoopAnalysis>(F),
          FAM.getResult<llvm::ScalarEvolutionAnalysis>(F));
      // The loop and SCEV results of this look must not reach the unroll
      // below (it would see trip counts from before loop rotation).
      FAM.invalidate(F, llvm::PreservedAnalyses::none());
      if (runtimeTrip && propagateCallSiteConstants(F)) {
        post.run(F, FAM);
        if (HLSAutoInlineReport)
          llvm::errs() << "hls-auto-inline: '" << F.getName()
                       << "': parameters every call passes as one constant "
                          "are propagated before measuring\n";
      }
    }
    if (foldConstantBuiltinStandIns(F))
      post.run(F, FAM);
    if (markUnrollLoops(F, FAM.getResult<llvm::LoopAnalysis>(F))) {
      llvm::FunctionPassManager prepare;
      prepare.addPass(llvm::LoopSimplifyPass());
      prepare.addPass(llvm::LCSSAPass());
      prepare.addPass(llvm::createFunctionToLoopPassAdaptor(
          llvm::LoopRotatePass(), /*UseMemorySSA=*/false));
      prepare.run(F, FAM);
      FunctionSize estimate =
          estimateUnrolled(F, FAM.getResult<llvm::LoopAnalysis>(F),
                           FAM.getResult<llvm::ScalarEvolutionAnalysis>(F),
                           budget);
      if (static_cast<uint64_t>(estimate.size) > budget) {
        if (HLSAutoInlineReport)
          llvm::errs() << "hls-auto-inline: '" << F.getName()
                       << "' is not unrolled for measuring: over "
                       << budget << " instructions after the unroll; size "
                       << "and calls are estimated\n";
        estimate.estimated = true;
        estimates[F.getName().str()] = std::move(estimate);
        continue;
      }
      llvm::FunctionPassManager unroll;
      unroll.addPass(llvm::LoopUnrollPass(llvm::LoopUnrollOptions(
          /*OptLevel=*/2, /*OnlyWhenForced=*/true, /*ForgetSCEV=*/false)));
      unroll.run(F, FAM);
    }
    post.run(F, FAM);
    while (foldConstantBuiltinStandIns(F))
      post.run(F, FAM);
    if (HLSAutoInlineRotatedSize)
      rotated.run(F, FAM);
    if (!HLSAutoInlineDump.empty() &&
        F.getName().contains(HLSAutoInlineDump))
      F.print(llvm::errs());
  }

  auto &functions = state().functions;
  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;
    FunctionSize fs;
    auto est = estimates.find(F.getName().str());
    if (est != estimates.end())
      fs = std::move(est->second);
    fs.returnsPointer = F.getReturnType()->isPointerTy();
    fs.alwaysInline = F.hasFnAttribute(llvm::Attribute::AlwaysInline);
    if (fs.estimated) {
      functions[F.getName().str()] = std::move(fs);
      continue;
    }
    if (HLSAutoInlineRotatedSize) {
      // N131: LLVM 7's rotation keeps each guarded loop's preheader / exit
      // block (one `br` each) and the block between two consecutive loops,
      // which LLVM 18's CFG cleanup folds: one instruction per loop with a
      // variable trip count, and per group of sibling loops one per
      // constant-trip loop after the first (kernel w9r17p_tc_a; 145 of 150
      // plain-C functions exact). The sibling term counts only the loops at
      // the top of the function and those directly in a variable-trip loop:
      // constant-trip loops nested in a constant-trip loop add nothing
      // (W9 round 17 review, kernels w9r17rev_rv_rot_nest51, w9r17fx_nest /
      // nest2: 10 of 12 nested functions exact, every flat one unchanged;
      // counting every constant loop of the function got 2 of 12, e.g.
      // rv_rot_nest51 `blk` 56 for Vitis' 54, kept where Vitis inlines).
      auto &LI = FAM.getResult<llvm::LoopAnalysis>(F);
      auto &SE = FAM.getResult<llvm::ScalarEvolutionAnalysis>(F);
      auto isConstantTrip = [&](llvm::Loop *L) {
        return SE.getSmallConstantTripCount(L) > 0;
      };
      auto siblingTerm = [&](llvm::ArrayRef<llvm::Loop *> siblings) {
        int64_t constant = llvm::count_if(siblings, isConstantTrip);
        return std::max<int64_t>(0, constant - 1);
      };
      int64_t extra = siblingTerm(LI.getTopLevelLoops());
      for (llvm::Loop *L : LI.getLoopsInPreorder())
        if (!isConstantTrip(L))
          extra += 1 + siblingTerm(L->getSubLoops());
      fs.size += extra;
    }
    for (llvm::Instruction &I : llvm::instructions(F)) {
      if (llvm::isa<llvm::DbgInfoIntrinsic>(I))
        continue;
      if (HLSAutoInlineRotatedSize && I.isLifetimeStartOrEnd())
        continue;
      // Integer casts are free in Vitis's count (N18: ai_gep_u16_e2_v2 116
      // with 16 index zext inlined, ai_gep_u16_e3_v2 118 kept).
      if (llvm::isa<llvm::SExtInst, llvm::ZExtInst, llvm::TruncInst>(I))
        continue;
      ++fs.size;
      // Vitis' front end does not contract `a * b + c` (no llvm.fmuladd in
      // any of the ground-truth runs' a.g.ld.5.gdce.ll), clang 18 does: an
      // fmuladd counts as Vitis' fmul + fadd (spector_hls dct `DCT8_auto`:
      // Vitis 56, kept; counted as one, 50 x 2 calls fell under the
      // threshold; W9 round 17 review fixes).
      if (HLSAutoInlineRotatedSize)
        if (auto *intr = llvm::dyn_cast<llvm::IntrinsicInst>(&I))
          if (intr->getIntrinsicID() == llvm::Intrinsic::fmuladd)
            ++fs.size;
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&I))
        if (llvm::Function *callee = call->getCalledFunction())
          if (countsAsDesignCall(callee))
            ++fs.calls[callee->getName().str()];
    }
    functions[F.getName().str()] = std::move(fs);
  }
  state().measured = true;
}

/// What the CodeGenerator refers to. A base class of InlineSizeConsumer
/// placed before the multiplexer that owns the CodeGenerator, so that it is
/// built before and destroyed after it.
struct SizeResources {
  llvm::LLVMContext context;
  clang::CodeGenOptions codeGenOpts;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags;
  clang::CodeGenerator *generator = nullptr;

  explicit SizeResources(clang::CompilerInstance &CI)
      : codeGenOpts(CI.getCodeGenOpts()) {
    // N18's proxy: clang -O0 -Xclang -disable-O0-optnone; locations only, so
    // that the loop metadata carries the loops' start locations.
    codeGenOpts.OptimizationLevel = 0;
    codeGenOpts.DisableO0ImplyOptNone = true;
    codeGenOpts.setDebugInfo(llvm::codegenoptions::LocTrackingOnly);
    codeGenOpts.DebugColumnInfo = true;
    // Its own diagnostics: whatever clang's codegen has to say must not
    // change cgeist's result.
    diags = new clang::DiagnosticsEngine(
        new clang::DiagnosticIDs(), new clang::DiagnosticOptions(),
        new clang::IgnoringDiagConsumer(), /*ShouldOwnClient=*/true);
  }
};

/// clang's CodeGenerator for the translation unit; when the translation unit
/// ends, its module is measured (cgeist's own consumer comes first, so the
/// unroll pragmas are recorded by then).
class InlineSizeConsumer : private SizeResources,
                           public clang::MultiplexConsumer {
public:
  InlineSizeConsumer(clang::CompilerInstance &CI, llvm::StringRef inFile)
      : SizeResources(CI), MultiplexConsumer(makeGenerator(*this, CI, inFile)),
        sourceDiags(CI.getDiagnostics()) {}

  // clang's CodeGenerator stops at the first error of its own diagnostics
  // engine, which here is a private one; it must also stop at the first
  // error in the source, or it generates code for an invalid AST and
  // asserts (an `undeclared identifier` became a crash). cgeist fails on
  // such a source anyway.
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
    return sourceFailed() || MultiplexConsumer::HandleTopLevelDecl(D);
  }
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleInlineFunctionDefinition(D);
  }
  void HandleInterestingDecl(clang::DeclGroupRef D) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleInterestingDecl(D);
  }
  void HandleTagDeclDefinition(clang::TagDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleTagDeclDefinition(D);
  }
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleTagDeclRequiredDefinition(D);
  }
  void HandleCXXImplicitFunctionInstantiation(clang::FunctionDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleCXXImplicitFunctionInstantiation(D);
  }
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleCXXStaticMemberVarInstantiation(D);
  }
  void CompleteTentativeDefinition(clang::VarDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::CompleteTentativeDefinition(D);
  }
  void CompleteExternalDeclaration(clang::VarDecl *D) override {
    if (!sourceFailed())
      MultiplexConsumer::CompleteExternalDeclaration(D);
  }
  void HandleVTable(clang::CXXRecordDecl *RD) override {
    if (!sourceFailed())
      MultiplexConsumer::HandleVTable(RD);
  }

  void HandleTranslationUnit(clang::ASTContext &C) override {
    if (sourceFailed())
      return;
    MultiplexConsumer::HandleTranslationUnit(C);
    if (diags->hasErrorOccurred()) {
      llvm::errs() << "warning: --hls-auto-inline: clang could not generate "
                      "LLVM IR for this translation unit; every call without "
                      "'inline off' is inlined\n";
      return;
    }
    std::unique_ptr<llvm::Module> M(generator->ReleaseModule());
    if (M)
      measure(*M);
  }

private:
  bool sourceFailed() const { return sourceDiags.hasErrorOccurred(); }

  clang::DiagnosticsEngine &sourceDiags;

  static std::vector<std::unique_ptr<clang::ASTConsumer>>
  makeGenerator(SizeResources &r, clang::CompilerInstance &CI,
                llvm::StringRef inFile) {
    std::unique_ptr<clang::CodeGenerator> gen(clang::CreateLLVMCodeGen(
        *r.diags, inFile, &CI.getVirtualFileSystem(), CI.getHeaderSearchOpts(),
        CI.getPreprocessorOpts(), r.codeGenOpts, r.context));
    r.generator = gen.get();
    std::vector<std::unique_ptr<clang::ASTConsumer>> v;
    v.push_back(std::move(gen));
    return v;
  }
};

//===----------------------------------------------------------------------===//
// The decision
//===----------------------------------------------------------------------===//

struct MarkAutoInlinePass
    : public mlir::PassWrapper<MarkAutoInlinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkAutoInlinePass)
  llvm::StringRef getArgument() const final { return "hls-mark-auto-inline"; }
  llvm::StringRef getDescription() const final {
    return "Mark the calls Vitis does not inline automatically";
  }
  void runOnOperation() final;
};

/// The trip count of `loop` when its `#pragma HLS unroll` is complete
/// (factor 0) and its bounds are constants: Vitis unrolls such a loop
/// ([HLS 214-186]) before it decides what to inline, so the calls in its body
/// are that many calls of the enclosing body. std::nullopt otherwise.
std::optional<int64_t> completeUnrollTrips(mlir::LoopLikeOpInterface loop) {
  // The pragmas are not attributes of the loop yet when the decision runs;
  // the loop is found by its source location (the `for` keyword, which
  // recordUnrollLoop recorded and cgeist's loop location carries).
  llvm::SmallVector<mlir::FileLineColLoc, 4> locs;
  loop->getLoc()->walk([&](mlir::Location l) {
    if (auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(l))
      locs.push_back(flc);
    return mlir::WalkResult::advance();
  });
  bool complete = llvm::any_of(state().unrollLoops, [&](const UnrollLoop &r) {
    return r.factor == 0 &&
           llvm::any_of(locs, [&](mlir::FileLineColLoc l) {
             return l.getLine() == r.line && l.getColumn() == r.col &&
                    llvm::sys::path::filename(l.getFilename().getValue()) ==
                        r.file;
           });
  });
  if (!complete)
    return std::nullopt;
  std::optional<mlir::OpFoldResult> lb = loop.getSingleLowerBound(),
                                    ub = loop.getSingleUpperBound(),
                                    st = loop.getSingleStep();
  if (!lb || !ub || !st)
    return std::nullopt;
  std::optional<int64_t> l = mlir::getConstantIntValue(*lb),
                         u = mlir::getConstantIntValue(*ub),
                         s = mlir::getConstantIntValue(*st);
  if (!l || !u || !s || *s <= 0)
    return std::nullopt;
  return *u > *l ? (*u - *l + *s - 1) / *s : 0;
}

/// Whether a loop of `fn` encloses `op` once Vitis has unrolled the loops
/// with a complete unroll pragma and constant bounds (completeUnrollTrips;
/// such loops are gone when Vitis looks for dataflow processes: kernel
/// w9r6rev n48a, four calls of `stage` in a fully unrolled loop of a
/// dataflow region give the four processes stage, stage_1..3, 4 BRAM, with
/// size 22 x 4 calls below the threshold). `copies` (optional) receives
/// the number of copies the unrolled loops make of `op`.
bool isInsideLoop(mlir::Operation *op, mlir::func::FuncOp fn,
                  int64_t *copies = nullptr) {
  int64_t k = 1;
  for (mlir::Operation *p = op->getParentOp(); p && p != fn.getOperation();
       p = p->getParentOp())
    if (auto loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(p)) {
      std::optional<int64_t> trips = completeUnrollTrips(loop);
      if (!trips)
        return true;
      k = std::min<int64_t>(k * *trips, int64_t(1) << 20);
    }
  if (copies)
    *copies = k;
  return false;
}

/// Calls `fn(call, insideLoop)` for every call that ends up in `caller` once
/// cgeist has inlined the always_inline callees: the calls of `caller`
/// itself and, transitively, those in the bodies of the always_inline
/// functions it calls (measure() counted them in `caller`). `insideLoop`:
/// a loop of `caller` or of an always_inline body on the way encloses it
/// (a completely unrolled loop with constant bounds does not count, and the
/// third argument is the number of copies its unrolling makes, else 1).
void forEachEffectiveCall(
    mlir::func::FuncOp caller,
    const std::function<mlir::func::FuncOp(llvm::StringRef)> &funcOf,
    const std::function<void(mlir::func::CallOp, bool, int64_t)> &fn) {
  const std::set<std::string> &alwaysInline = state().alwaysInline;
  std::set<std::string> onPath;
  std::function<void(mlir::func::FuncOp, bool)> walk =
      [&](mlir::func::FuncOp f, bool outerInLoop) {
        f.walk([&](mlir::func::CallOp call) {
          int64_t copies = 1;
          bool inLoop = outerInLoop || isInsideLoop(call, f, &copies);
          fn(call, inLoop, inLoop ? 1 : copies);
          std::string callee = call.getCallee().str();
          if (!alwaysInline.count(callee) || onPath.count(callee))
            return;
          if (mlir::func::FuncOp body = funcOf(callee)) {
            onPath.insert(callee);
            walk(body, inLoop);
            onPath.erase(callee);
          }
        });
      };
  if (caller)
    walk(caller, false);
}

llvm::StringRef displayName(mlir::func::FuncOp f, llvm::StringRef symbol) {
  if (f)
    if (auto n = f->getAttrOfType<mlir::StringAttr>("hls.orig_name"))
      return n.getValue();
  return symbol;
}

void MarkAutoInlinePass::runOnOperation() {
  AutoInlineState &st = state();
  if (!st.measured)
    return;
  mlir::ModuleOp module = getOperation();
  mlir::Builder b(module.getContext());

  llvm::StringMap<mlir::func::FuncOp> funcs;
  module.walk([&](mlir::func::FuncOp f) {
    if (!f.isExternal())
      funcs[f.getSymName()] = f;
  });
  std::function<mlir::func::FuncOp(llvm::StringRef)> funcOf =
      [&](llvm::StringRef name) -> mlir::func::FuncOp {
    auto it = funcs.find(name);
    return it == funcs.end() ? mlir::func::FuncOp() : it->second;
  };

  /// Which calls of a (caller, callee) pair are kept.
  enum class Which { All, OutsideLoops, InsideLoops };
  struct Kept {
    std::string caller, callee;
    int64_t calls, size;
    Which which;
  };
  std::vector<Kept> kept;
  std::map<std::string, int64_t> finalSize;
  std::set<std::string> visiting, done;
  const int64_t threshold = HLSAutoInlineThreshold;

  // The calls of `callee` in `callerOp`, split by whether a loop of the
  // caller encloses them.
  auto countCalls = [&](mlir::func::FuncOp callerOp, llvm::StringRef callee,
                        int64_t &outside, int64_t &inside) {
    outside = inside = 0;
    forEachEffectiveCall(callerOp, funcOf,
                         [&](mlir::func::CallOp call, bool inLoop,
                             int64_t copies) {
                           if (call.getCallee() == callee)
                             (inLoop ? inside : outside) += copies;
                         });
  };

  // Bottom-up over the call graph: a callee is decided in its own body
  // (and its size grows by what it inlines) before its callers look at it.
  std::function<void(const std::string &)> visit = [&](const std::string &c) {
    if (done.count(c) || visiting.count(c))
      return;
    auto it = st.functions.find(c);
    if (it == st.functions.end())
      return;
    visiting.insert(c);
    for (auto &[callee, n] : it->second.calls)
      visit(callee);
    mlir::func::FuncOp callerOp = funcOf(c);
    // An always_inline function the always-inliner kept (e.g. external
    // linkage) is inlined into its callers anyway; their decisions cover its
    // calls (forEachEffectiveCall), so it marks none itself.
    const bool callerInlinesAway = it->second.alwaysInline;
    const bool callerIsDataflow =
        HLSAutoInlineDataflow && callerOp && callerOp->hasAttr("hls.dataflow");
    int64_t size = it->second.size;
    for (const auto &entry : it->second.calls) {
      const std::string &callee = entry.first;
      const int64_t n = entry.second;
      auto calleeInfo = st.functions.find(callee);
      if (calleeInfo == st.functions.end() || !finalSize.count(callee))
        continue; // recursion (Vitis: an error, 214-139): leave to the inliner
      mlir::func::FuncOp calleeOp = funcOf(callee);
      int64_t calleeSize = finalSize[callee];
      auto report = [&](const char *decision, const char *reason, int64_t k) {
        if (HLSAutoInlineReport)
          llvm::errs() << "hls-auto-inline: caller '"
                       << displayName(callerOp, c) << "' callee '"
                       << displayName(calleeOp, callee) << "' size "
                       << calleeSize << " calls " << k << ": " << decision
                       << " (" << reason << ")\n";
      };
      auto inlineCalls = [&](const char *reason, int64_t k) {
        size += k * (calleeSize - 2);
        report("inlined", reason, k);
      };
      auto keepCalls = [&](const char *reason, int64_t k, Which which) {
        if (!callerInlinesAway)
          kept.push_back({c, callee, k, calleeSize, which});
        report("kept", reason, k);
      };
      if (calleeOp && calleeOp->hasAttr("hls.inline_off")) {
        report("kept", "inline off", n);
        continue;
      }
      if (calleeOp && calleeOp->hasAttr("hls.inline")) {
        inlineCalls("inline pragma", n);
        continue;
      }
      // always_inline: clang inlines it before Vitis's own inliner runs; the
      // Vitis headers mark every ap_int / ap_fixed member and operator so
      // under __SYNTHESIS__ (AP_INLINE, etc/ap_common.h), and none of them
      // appears in a [HLS 214-178] line of the 208 corpus logs that use them.
      // measure() has inlined them already; this catches the calls the
      // always-inliner leaves (recursion).
      if (calleeInfo->second.alwaysInline) {
        inlineCalls("always_inline", n);
        continue;
      }
      if (HLSAutoInlineDataflow && calleeOp &&
          calleeOp->hasAttr("hls.dataflow")) {
        keepCalls("holds a dataflow region", n, Which::All);
        continue;
      }
      // A call at the top level of a dataflow region is a process, kept
      // whatever its size, also once a completely unrolled loop of the
      // region has made it one (isInsideLoop); a call inside another loop of
      // the region is not a process of its own and follows the size rule.
      int64_t rest = n;
      Which restWhich = Which::All;
      if (callerIsDataflow) {
        int64_t outside = 0, inside = 0;
        countCalls(callerOp, callee, outside, inside);
        if (outside > 0) {
          keepCalls("dataflow process", outside, Which::OutsideLoops);
          rest = inside > 0 ? std::max<int64_t>(n - outside, inside) : 0;
          restWhich = Which::InsideLoops;
          if (rest == 0)
            continue;
        }
      }
      if (calleeInfo->second.returnsPointer)
        inlineCalls("returns a pointer", rest);
      else if (calleeSize * rest > threshold)
        keepCalls("size x calls > threshold", rest, restWhich);
      else
        inlineCalls("size x calls <= threshold", rest);
    }
    finalSize[c] = size;
    visiting.erase(c);
    done.insert(c);
  };
  for (auto &[name, fs] : st.functions)
    visit(name);

  // A kept call that sits in the body of an always_inline function is marked
  // there, so it is kept in every caller of that function: a caller that
  // would inline it is not modelled (keep wins; no corpus case known).
  for (const Kept &k : kept) {
    mlir::func::FuncOp callerOp = funcOf(k.caller), calleeOp = funcOf(k.callee);
    if (!callerOp || !calleeOp)
      continue;
    bool marked = false;
    forEachEffectiveCall(callerOp, funcOf, [&](mlir::func::CallOp call,
                                               bool inLoop, int64_t) {
      if (call.getCallee() != k.callee)
        return;
      if (k.which != Which::All &&
          inLoop != (k.which == Which::InsideLoops))
        return;
      call->setAttr(kInlineAutoOffAttr,
                    b.getDictionaryAttr(
                        {b.getNamedAttr("calls", b.getI64IntegerAttr(k.calls)),
                         b.getNamedAttr("size", b.getI64IntegerAttr(k.size))}));
      marked = true;
    });
    if (marked)
      calleeOp->setAttr(kInlineAutoOffAttr,
                        b.getDictionaryAttr({b.getNamedAttr(
                            "size", b.getI64IntegerAttr(k.size))}));
  }
}

} // namespace

bool cgeist::hls::autoInlineEnabled() {
  return HLSAutoInline && isEnabled() && !checkOnly();
}

static std::optional<LoopLoc> loopLoc(const clang::SourceManager &SM,
                                      clang::SourceLocation loop) {
  if (!autoInlineEnabled() || loop.isInvalid())
    return std::nullopt;
  clang::PresumedLoc p = SM.getPresumedLoc(loop);
  if (p.isInvalid())
    return std::nullopt;
  return LoopLoc{llvm::sys::path::filename(p.getFilename()).str(),
                 p.getLine(), p.getColumn()};
}

void cgeist::hls::recordUnrollLoop(const clang::SourceManager &SM,
                                   clang::SourceLocation loop, int64_t factor) {
  if (std::optional<LoopLoc> l = loopLoc(SM, loop)) {
    UnrollLoop r;
    static_cast<LoopLoc &>(r) = *l;
    r.factor = factor;
    state().unrollLoops.push_back(std::move(r));
  }
}

void cgeist::hls::recordPipelineLoop(const clang::SourceManager &SM,
                                     clang::SourceLocation loop) {
  if (std::optional<LoopLoc> l = loopLoc(SM, loop))
    state().pipelineLoops.push_back(std::move(*l));
}

void cgeist::hls::recordPipelineFunction(llvm::StringRef symbol) {
  if (autoInlineEnabled())
    state().pipelineFunctions.insert(symbol.str());
}

std::unique_ptr<clang::ASTConsumer>
cgeist::hls::createInlineSizeConsumer(clang::CompilerInstance &CI,
                                      llvm::StringRef inFile) {
  return std::make_unique<InlineSizeConsumer>(CI, inFile);
}

std::unique_ptr<mlir::Pass> cgeist::hls::createMarkAutoInlinePass() {
  return std::make_unique<MarkAutoInlinePass>();
}
