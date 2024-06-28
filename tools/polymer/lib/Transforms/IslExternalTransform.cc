
#include "mlir/IR/Verifier.h"
#include "polymer/Support/IslScop.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/ISL.h"
#include "polymer/Transforms/ExtractScopStmt.h"
#include "polymer/Transforms/PlutoTransform.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/id.h"
#include "isl/id_to_id.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/val.h"

using namespace mlir;
using namespace polymer;

using llvm::dbgs;

#define DEBUG_TYPE "islexternal-opt"

static llvm::cl::opt<std::string> ClIslExternalDumpSchedules(
    "islexternal-dump-schedules",
    llvm::cl::desc("Directory to dump ISL schedules to"));

static llvm::cl::opt<std::string> ClIslExternalDumpAccesses(
    "islexternal-dump-accesses",
    llvm::cl::desc("Directory to dump ISL accesses to"));

static llvm::cl::opt<std::string> ClIslExternalImportSchedules(
    "islexternal-import-schedules",
    llvm::cl::desc("Directory to import ISL schedules from"));

namespace polymer {

mlir::func::FuncOp islexternalTransform(mlir::func::FuncOp f,
                                        OpBuilder &rewriter) {
  LLVM_DEBUG(dbgs() << "IslExternal transforming: \n");
  LLVM_DEBUG(f.dump());

  StringRef funcName = f.getName();

  std::unique_ptr<IslScop> scop = createIslFromFuncOp(f);

  std::error_code EC;
  if (ClIslExternalDumpSchedules.getNumOccurrences() > 0) {
    std::string fname = (ClIslExternalDumpSchedules + "/" + funcName).str();
    llvm::raw_fd_ostream ScheduleOut(fname, EC);
    if (EC) {
      llvm::errs() << "Can't open " << fname << "\n";
      abort();
    }
    scop->dumpSchedule(ScheduleOut);
  }
  if (ClIslExternalDumpAccesses.getNumOccurrences() > 0) {
    std::string fname = (ClIslExternalDumpAccesses + "/" + funcName).str();
    llvm::raw_fd_ostream AccessesOut(fname, EC);
    if (EC) {
      llvm::errs() << "Can't open " << fname << "\n";
      abort();
    }
    scop->dumpAccesses(AccessesOut);
  }

  isl_schedule *newSchedule = nullptr;
  if (ClIslExternalImportSchedules.getNumOccurrences() == 0) {
    // Do a round trip
    newSchedule = isl_schedule_copy(scop->getSchedule());
  } else {
    std::string fname = (ClIslExternalImportSchedules + "/" + funcName).str();
    auto ScheduleIn = llvm::MemoryBuffer::getFileAsStream(fname);
    if (std::error_code EC = ScheduleIn.getError()) {
      llvm::errs() << "Can't open " << fname << "\n";
      abort();
    }
    newSchedule =
        isl_schedule_read_from_str(isl_schedule_get_ctx(scop->getSchedule()),
                                   ScheduleIn->get()->getBufferStart());
    if (!newSchedule) {
      llvm::errs() << "Could not read valid isl schedule from " << fname
                   << "\n";
      abort();
    }
  }
  assert(newSchedule);

  mlir::func::FuncOp g = scop->applySchedule(newSchedule, f);
  assert(mlir::verify(g).succeeded());

  if (g) {
    SmallVector<DictionaryAttr> argAttrs;
    f.getAllArgAttrs(argAttrs);
    g.setAllArgAttrs(argAttrs);
  }

  return g;
}

mlir::func::FuncOp plutoTransform(mlir::func::FuncOp f,
                                  mlir::OpBuilder &rewriter,
                                  std::string dumpClastAfterPluto,
                                  bool parallelize = false, bool debug = false,
                                  int cloogf = -1, int cloogl = -1,
                                  bool diamondTiling = false) {
  llvm_unreachable("not compiled with pluto support");
}
void registerPlutoTransformPass() {}
} // namespace polymer
