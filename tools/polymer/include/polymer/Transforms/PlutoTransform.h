//===- PlutoTransform.h - Transform MLIR code by PLUTO --------------------===//
//
// This file declares the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TRANSFORMS_PLUTOTRANSFORM_H
#define POLYMER_TRANSFORMS_PLUTOTRANSFORM_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace polymer {

struct PlutoOptPipelineOptions
    : public mlir::PassPipelineOptions<PlutoOptPipelineOptions> {
  Option<std::string> dumpClastAfterPluto{
      *this, "dump-clast-after-pluto",
      llvm::cl::desc("File name for dumping the CLooG AST (clast) after Pluto "
                     "optimization.")};
  Option<bool> parallelize{*this, "parallelize",
                           llvm::cl::desc("Enable parallelization from Pluto."),
                           llvm::cl::init(false)};
  Option<bool> debug{*this, "debug",
                     llvm::cl::desc("Enable moredebug in Pluto."),
                     llvm::cl::init(false)};
  Option<bool> generateParallel{
      *this, "gen-parallel", llvm::cl::desc("Generate parallel affine loops."),
      llvm::cl::init(false)};

  Option<int> cloogf{*this, "cloogf", llvm::cl::desc("-cloogf option."),
                     llvm::cl::init(-1)};
  Option<int> cloogl{*this, "cloogl", llvm::cl::desc("-cloogl option."),
                     llvm::cl::init(-1)};
  Option<bool> diamondTiling{*this, "diamond-tiling",
                             llvm::cl::desc("Enable diamond tiling"),
                             llvm::cl::init(false)};
};

void registerPlutoTransformPass();
void addPlutoOpt(mlir::OpPassManager &pm,
                 const PlutoOptPipelineOptions &pipelineOptions);
} // namespace polymer

#endif
