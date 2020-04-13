#include "ctx.h"
#include "islAst.h"
#include "islNodeBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlirCodegen.h"
#include "scop.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <fstream>

using namespace llvm;
static cl::opt<std::string> outputFileName("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("out"));
static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<Specify input file>"),
                                          cl::Required);

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false));

int main(int argc, char **argv) {

  using namespace mlir;
  using namespace pet;
  using namespace util;
  using namespace ast;
  using namespace codegen;

  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream inputFile(inputFileName);
  if (!inputFile.good()) {
    outs() << "Not able to open file: " << inputFileName << "\n";
    return -1;
  }

  auto ctx = ScopedCtx();
  auto petScop = Scop::parseFile(ctx, inputFileName);
  // petScop.dump();

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
