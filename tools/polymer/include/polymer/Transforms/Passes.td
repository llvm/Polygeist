#ifndef POLYMER_TRANSFORMS_PASSES
#define POLYMER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def AnnotateScop : FunctionPass<"annotate-scop"> {
  let summary = "Annotate scop.ignore to unselected functions.";
  let constructor = "polymer::createAnnotateScopPass()";

  let options = [
    ListOption<
      "includedFunctions",
      "functions",
      "std::string",   
      "A list of functions to be included.",
      "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

#endif
