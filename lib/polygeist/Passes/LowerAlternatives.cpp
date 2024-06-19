#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Passes/Passes.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>

#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"

using namespace mlir;
using namespace polygeist;

llvm::cl::opt<PolygeistAlternativesMode> PolygeistAlternativesMode(
    "polygeist-alternatives-mode", llvm::cl::init(PAM_Static),
    llvm::cl::desc("Polygeist alternatives op mode"),
    llvm::cl::values(
        clEnumValN(PAM_Static, "static", "Pick at compile time"),
        clEnumValN(PAM_PGO_Profile, "pgo_prof",
                   "Profile Guided Optimization - profiling mode"),
        clEnumValN(PAM_PGO_Opt, "pgo_opt",
                   "Profile Guided Optimization - optimization mode")));

namespace {

struct LowerGPUAlternativesOp
    : public OpRewritePattern<polygeist::AlternativesOp> {
  using OpRewritePattern<polygeist::AlternativesOp>::OpRewritePattern;
  const char *PATTERN = "lower-gpu-alternatives";
  LogicalResult matchAndRewrite(polygeist::AlternativesOp gao,
                                PatternRewriter &rewriter) const override {

    if (gao->getAttrOfType<StringAttr>("alternatives.type").getValue() !=
        "gpu_kernel")
      return failure();

    auto locStr = gao->getAttrOfType<StringAttr>("polygeist.altop.id").data();

    auto descs = gao->getAttrOfType<ArrayAttr>("alternatives.descs");

    if (PolygeistAlternativesMode == PAM_PGO_Opt) {
      std::string dirname = []() {
        if (char *d = getenv(POLYGEIST_PGO_DATA_DIR_ENV_VAR)) {
          return std::string(d);
        } else {
          return std::string(POLYGEIST_PGO_DEFAULT_DATA_DIR);
        }
      }();
      // TODO error handling
      std::ifstream ifile;
      int numAlternatives = gao->getNumRegions();
      std::vector<std::vector<double>> timings;
      for (int i = 0; i < numAlternatives; i++) {
        timings.push_back({});
      }
      ifile.open(std::string(dirname) + "/" + locStr, std::ios::in);
      while (ifile) {
        int alt;
        double time;
        ifile >> alt >> time;
        if (alt >= 0 && alt < numAlternatives) {
          timings[alt].push_back(time);
        } else {
          llvm::errs() << "Invalid alternative data";
          assert(0);
        }
      }
      std::vector<double> avgs;
      for (int i = 0; i < numAlternatives; i++) {
        if (timings[i].size() == 0) {
          llvm::errs() << "No data for alternative " << i << "," << descs[i]
                       << " of " << locStr << "\n";
          assert(0);
          avgs.push_back(std::numeric_limits<double>::infinity());
        } else {
          // TODO might get some round off errors here, maybe use a better alg
          // or median
          avgs.push_back(
              std::accumulate(timings[i].begin(), timings[i].end(), 0.0f) /
              timings[i].size());
          llvm::errs() << "Alternative " << i << "," << descs[i] << " is "
                       << avgs[i] << "\n";
        }
      }

      int bestAlt = std::distance(avgs.begin(),
                                  std::min_element(avgs.begin(), avgs.end()));
      llvm::errs() << "Picking " << bestAlt << "," << descs[bestAlt] << "\n";

      auto block = &*gao->getRegions()[bestAlt].begin();

      rewriter.eraseOp(block->getTerminator());
      rewriter.inlineBlockBefore(block, gao);
      rewriter.eraseOp(gao);

      return success();
    } else {
      llvm_unreachable("Invalid enum");
    }
  }
};
} // namespace

struct LowerAlternativesPass
    : public LowerAlternativesBase<LowerAlternativesPass> {
  void runOnOperation() override {
    if (char *e = getenv("POLYGEIST_CHOOSE_ALTERNATIVE")) {
      int id = atoi(e);

      std::vector<polygeist::AlternativesOp> toHandle;
      getOperation()->walk(
          [&](polygeist::AlternativesOp aop) { toHandle.push_back(aop); });
      for (auto aop : toHandle) {
        if (id == -1)
          id = aop->getNumRegions() - 1;
        if (id < 0 || (unsigned)id >= aop->getNumRegions()) {
          llvm::errs() << "Invalid alternative ID " << id << "\n";
          return;
        }
        auto block = &*aop->getRegions()[id].begin();

        block->getTerminator()->erase();
        OpBuilder builder(aop);
        IRMapping mapping;
        for (auto &op : *block) {
          builder.clone(op, mapping);
        }
        aop->erase();
      }
      return;
    }

    // TODO Should be its own pass really
    std::map<std::string, int> num;
    getOperation()->walk([&](polygeist::AlternativesOp altOp) {
      std::string funcName;
      if (auto funcOp = altOp->getParentOfType<LLVM::LLVMFuncOp>()) {
        funcName = funcOp.getName();
        funcName += ".llvm";
      } else if (auto funcOp = altOp->getParentOfType<func::FuncOp>()) {
        funcName = funcOp.getName();
        funcName += ".func";
      } else {
        llvm_unreachable("How?");
      }
      if (num.count(funcName) == 0)
        num[funcName] = 0;
      std::string id = funcName + "." + std::to_string(num[funcName]++);

      Location loc = altOp->getLoc();
      std::string locStr = [&loc]() {
        std::string str;
        llvm::raw_string_ostream stream(str);
        loc.print(stream);
        stream.flush();
        return stream.str();
      }();
      locStr += id;
      static std::string cwd = std::filesystem::current_path().string();
      locStr = cwd + locStr;
      for (char &c : locStr)
        if (c == '/')
          c = '+';
      altOp->setAttr("polygeist.altop.id",
                     StringAttr::get(&getContext(), locStr));
    });

    if (PolygeistAlternativesMode == PAM_PGO_Opt) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<LowerGPUAlternativesOp>(&getContext());
      GreedyRewriteConfig config;
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<Pass> mlir::polygeist::createLowerAlternativesPass() {
  return std::make_unique<LowerAlternativesPass>();
}
