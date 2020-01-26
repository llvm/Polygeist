#include "ctx.h"
#include "islAst.h"
#include "islNodeBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlirCodegen.h"
#include "scop.h"
#include <iostream>

int main() {

  using namespace mlir;
  using namespace pet;
  using namespace util;
  using namespace ast;
  using namespace codegen;

  auto ctx = ScopedCtx();
  auto petScop = Scop::parseFile(
      ctx,
      "/home/parallels/mlir/llvm-project/mlir/tools/mlir-pet/inputs/gemm.c");
  // petScop.dump();

  MLIRContext context;
  MLIRCodegen MLIRbuilder(context, petScop);

  auto ISLAst = IslAst(petScop);
  // ISLAst.dump();

  auto ISLNodeBuilder = IslNodeBuilder(ISLAst, MLIRbuilder);
  ISLNodeBuilder.MLIRFromISLAst();
  MLIRbuilder.dump();
  return 0;
}
