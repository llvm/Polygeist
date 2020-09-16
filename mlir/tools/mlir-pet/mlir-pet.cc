#include "Lib/ctx.h"
#include "Lib/islAst.h"
#include "Lib/islNodeBuilder.h"
#include "Lib/mlirCodegen.h"
#include "Lib/scop.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <fstream>
#include <iostream>

using namespace llvm;

static cl::OptionCategory toolOptions("pet to mlir - tool options");

static cl::opt<std::string> outputFileName("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("out"),
                                           cl::cat(toolOptions));

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<Specify input file>"),
                                          cl::Required, cl::cat(toolOptions));


static cl::opt<std::string> cfunction(cl::Positional,
                                          cl::desc("<Specify function>"),
                                          cl::Required, cl::cat(toolOptions));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::opt<bool> reschedule("reschedule",
                                llvm::cl::desc("Reschedule with ISL"),
                                llvm::cl::init(false), cl::cat(toolOptions));

static cl::opt<bool> dumpSchedule("dump-schedule",
                                  llvm::cl::desc("Pretty print the schedule"),
                                  llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));
// check if the schedule is bounded.
static bool isUnbounded(isl::schedule schedule) {
  auto isUnBoundedSet = [](isl::set set) -> bool {
    return !(set.is_bounded());
  };
  std::vector<isl::set> domains;
  schedule.get_domain().foreach_set([&](isl::set set) {
    domains.push_back(set);
    return isl_stat_ok;
  });
  int unBounded = count_if(domains.begin(), domains.end(), isUnBoundedSet);
  return unBounded != 0;
}

static void dumpScheduleWithIsl(isl::schedule schedule, llvm::raw_ostream &os) {
  auto ctx = schedule.get_ctx().get();
  auto *p = isl_printer_to_str(ctx);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule(p, schedule.get());
  auto *str = isl_printer_get_str(p);
  os << str << "\n";
  free(str);
  isl_printer_free(p);
}

static isl::schedule rescheduleWithIsl(pet::Scop &scop) {
  auto proximity = scop.getAllDependences();
  auto validity = scop.getAllDependences();

  auto sc = isl::schedule_constraints::on_domain(scop.getNonKilledDomain());

  sc = sc.set_proximity(proximity);
  sc = sc.set_validity(validity);
  sc = sc.set_coincidence(validity);
  auto schedule = sc.compute_schedule();
  return schedule;
}

#include "Lib/clang-mlir.cc"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
int main(int argc, char **argv) {

  using namespace mlir;
  using namespace pet;
  using namespace util;
  using namespace ast;
  using namespace codegen;

  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream inputFile(inputFileName);
  if (!inputFile.good()) {
    outs() << "Not able to open file: " << inputFileName << "\n";
    return -1;
  }

  //registerDialect<AffineDialect>();
  //registerDialect<StandardOpsDialect>();
  MLIRContext context;

  context.getOrLoadDialect<AffineDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  //MLIRContext context;

  if (showDialects) {
    outs() << "Registered Dialects:\n";
    for (Dialect *dialect : context.getLoadedDialects()) {
      outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }
  MLIRCodegen MLIRbuilder(context);

  parseMLIR(inputFileName.c_str(), cfunction, includeDirs, MLIRbuilder);

  mlir::PassManager pm(&context);

    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createParallelLowerPass());


  if (mlir::failed(pm.run(MLIRbuilder.theModule_)))
    return 4;

  MLIRbuilder.print(outs());
  return 0;
}

// ./clang -O3 -mllvm -polly -c test.c -mllvm -polly-process-unprofitable -mllvm -polly-export