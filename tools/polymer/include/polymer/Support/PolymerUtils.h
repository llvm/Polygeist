#ifndef POLYMER_SUPPORT_POLYMERUTILS_H_
#define POLYMER_SUPPORT_POLYMERUTILS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace polymer {
mlir::func::FuncOp islexternalTransform(mlir::func::FuncOp f,
                                        mlir::OpBuilder &rewriter);
mlir::func::FuncOp plutoTransform(mlir::func::FuncOp f,
                                  mlir::OpBuilder &rewriter,
                                  std::string dumpClastAfterPluto,
                                  bool parallelize = false, bool debug = false,
                                  int cloogf = -1, int cloogl = -1,
                                  bool diamondTiling = false);
unsigned extractScopStmt(mlir::func::FuncOp f, mlir::OpBuilder &b);
void replaceUsesByStored(mlir::func::FuncOp f, mlir::OpBuilder &b);
void separateAffineIfBlocks(mlir::func::FuncOp f, mlir::OpBuilder &b);
void demoteRegisterToMemory(mlir::func::FuncOp f, mlir::OpBuilder &b);
void dedupIndexCast(mlir::func::FuncOp f);
void plutoParallelize(mlir::func::FuncOp f, mlir::OpBuilder b);
void demoteLoopReduction(mlir::func::FuncOp f, mlir::OpBuilder &b);
void demoteLoopReduction(mlir::func::FuncOp f, mlir::affine::AffineForOp forOp,
                         mlir::OpBuilder &b);
} // namespace polymer

#endif // POLYMERUTILS_H_
