#include "ctx.h"
#include "islAst.h"
#include "islNodeBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlirCodegen.h"
#include "scop.h"
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

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

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

  // pass include paths to pet.
  struct pet_options *options;
  options = pet_options_new_with_defaults();
  std::vector<char *> arguments;
  char argument1[] = "program";
  char argumentI[] = "-I";
  arguments.push_back(argument1);
  for (const auto &includePath : includeDirs) {
    arguments.push_back(argumentI);
    arguments.push_back(const_cast<char *>(includePath.c_str()));
  }
  int argsCount = arguments.size();
  argsCount = pet_options_parse(options, argsCount, &arguments[0], ISL_ARG_ALL);
  auto ctx = ScopedCtx(isl_ctx_alloc_with_options(&pet_options_args, options));

  auto petScop = Scop::parseFile(ctx, inputFileName);
  // petScop.dump();

  // check if the schedule is bounded.
  auto isUnBounded = [](isl::set set) -> bool { return !(set.is_bounded()); };

  std::vector<isl::set> domains;
  auto schedule = petScop.getSchedule();
  schedule.get_domain().foreach_set([&](isl::set set) {
    domains.push_back(set);
    return isl_stat_ok;
  });
  int unBounded = count_if(domains.begin(), domains.end(), isUnBounded);
  if (unBounded != 0) {
    outs() << "schedule must be bounded\n";
    return -1;
  }

  registerDialect<AffineDialect>();
  registerDialect<StandardOpsDialect>();
  MLIRContext context;
  if (showDialects) {
    outs() << "Registered Dialects:\n";
    for (Dialect *dialect : context.getRegisteredDialects()) {
      outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }
  MLIRCodegen MLIRbuilder(context, petScop);

  auto ISLAst = IslAst(petScop);
  // ISLAst.dump();

  auto ISLNodeBuilder = IslNodeBuilder(ISLAst, MLIRbuilder);
  ISLNodeBuilder.MLIRFromISLAst();
  // MLIRbuilder.dump();

  if (outputFileName.empty()) {
    MLIRbuilder.print(outs());
    return 0;
  }

  std::error_code ec;
  ToolOutputFile out(outputFileName, ec, sys::fs::OF_None);
  if (ec) {
    WithColor::error() << ec.message() << "\n";
    return -1;
  }
  MLIRbuilder.print(out.os());
  out.keep();
  return 0;
}
