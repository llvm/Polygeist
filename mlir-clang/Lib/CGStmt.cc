//===- CGStmt.cc - Emit MLIR IRs by walking stmt-like AST nodes-*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "mlir/IR/Diagnostics.h"
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/SCF/SCF.h>

using namespace mlir;
using namespace mlir::arith;

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}

ValueCategory MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getForLoc());

  mlirclang::AffineLoopDescriptor affineLoopDescr;
  if (Glob.scopLocList.isInScop(fors->getForLoc()) &&
      isTrivialAffineLoop(fors, affineLoopDescr)) {
    buildAffineLoop(fors, loc, affineLoopDescr);
  } else {

    if (auto s = fors->getInit()) {
      Visit(s);
    }

    auto i1Ty = builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = builder.create<ConstantIntOp>(loc, true, 1);

    LoopContext lctx{builder.create<mlir::memref::AllocaOp>(loc, type),
                     builder.create<mlir::memref::AllocaOp>(loc, type)};
    builder.create<mlir::memref::StoreOp>(loc, truev, lctx.noBreak);

    auto toadd = builder.getInsertionBlock()->getParent();
    auto &condB = *(new Block());
    toadd->getBlocks().push_back(&condB);
    auto &bodyB = *(new Block());
    toadd->getBlocks().push_back(&bodyB);
    auto &exitB = *(new Block());
    toadd->getBlocks().push_back(&exitB);

    builder.create<mlir::BranchOp>(loc, &condB);

    builder.setInsertionPointToStart(&condB);

    if (auto s = fors->getCond()) {
      auto condRes = Visit(s);
      auto cond = condRes.getValue(builder);
      if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
        cond = builder.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
      }
      auto ty = cond.getType().cast<mlir::IntegerType>();
      if (ty.getWidth() != 1) {
        ty = builder.getIntegerType(1);
        cond = builder.create<arith::TruncIOp>(loc, cond, ty);
      }
      auto nb = builder.create<mlir::memref::LoadOp>(
          loc, lctx.noBreak, std::vector<mlir::Value>());
      cond = builder.create<AndIOp>(loc, cond, nb);
      builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
    }

    builder.setInsertionPointToStart(&bodyB);
    builder.create<mlir::memref::StoreOp>(
        loc,
        builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                             std::vector<mlir::Value>()),
        lctx.keepRunning, std::vector<mlir::Value>());

    loops.push_back(lctx);
    Visit(fors->getBody());
    if (auto s = fors->getInc()) {
      Visit(s);
    }
    loops.pop_back();
    if (builder.getInsertionBlock()->empty() ||
        !isTerminator(&builder.getInsertionBlock()->back())) {
      builder.create<mlir::BranchOp>(loc, &condB);
    }

    builder.setInsertionPointToStart(&exitB);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitDoStmt(clang::DoStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getDoLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::BranchOp>(loc, &bodyB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      cond = builder.create<arith::TruncIOp>(loc, cond, ty);
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitWhileStmt(clang::WhileStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getLParenLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      cond = builder.create<arith::TruncIOp>(loc, cond, ty);
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getIfLoc());
  auto cond = Visit(stmt->getCond()).getValue(builder);
  assert(cond != nullptr && "must be a non-null");

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    auto postTy = builder.getI1Type();
    cond = builder.create<arith::TruncIOp>(loc, cond, postTy);
  }
  bool hasElseRegion = stmt->getElse();
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond, hasElseRegion);

  ifOp.thenRegion().back().clear();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());
  Visit(stmt->getThen());
  builder.create<scf::YieldOp>(loc);
  if (hasElseRegion) {
    ifOp.elseRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    Visit(stmt->getElse());
    builder.create<scf::YieldOp>(loc);
  }

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitDeclStmt(clang::DeclStmt *decl) {
  IfScope scope(*this);
  for (auto sub : decl->decls()) {
    if (auto vd = dyn_cast<VarDecl>(sub)) {
      VisitVarDecl(vd);
    } else if (isa<TypeAliasDecl, RecordDecl, StaticAssertDecl, TypedefDecl,
                   UsingDecl>(sub)) {
    } else {
      emitError(getMLIRLocation(decl->getBeginLoc()))
          << " + visiting unknonwn sub decl stmt\n";
      sub->dump();
      assert(0 && "unknown sub decl");
    }
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitAttributedStmt(AttributedStmt *AS) {
  emitWarning(getMLIRLocation(AS->getAttrLoc())) << "ignoring attributes\n";
  return Visit(AS->getSubStmt());
}

ValueCategory MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto a : stmt->children()) {
    IfScope scope(*this);
    Visit(a);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  assert(loops.back().noBreak && "no break false");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().noBreak);

  return nullptr;
}

ValueCategory MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  return nullptr;
}

ValueCategory MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  IfScope scope(*this);
  bool isArrayReturn = false;
  Glob.getMLIRType(EmittingFunctionDecl->getReturnType(), &isArrayReturn);

  if (isArrayReturn) {
    auto rv = Visit(stmt->getRetValue());
    assert(rv.val && "expect right value to be valid");
    assert(rv.isReference && "right value must be a reference");
    auto op = function.getArgument(function.getNumArguments() - 1);
    assert(rv.val.getType().cast<MemRefType>().getElementType() ==
               op.getType().cast<MemRefType>().getElementType() &&
           "type mismatch");
    assert(op.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape()[1] ==
           op.getType().cast<MemRefType>().getShape()[1]);

    for (int i = 0; i < op.getType().cast<MemRefType>().getShape()[1]; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2);
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::memref::LoadOp>(loc, rv.val, idx), op, idx);
    }
  } else if (stmt->getRetValue()) {
    auto rv = Visit(stmt->getRetValue());
    if (!stmt->getRetValue()->getType()->isVoidType()) {
      if (!rv.val) {
        stmt->dump();
      }
      assert(rv.val && "expect right value to be valid");

      mlir::Value val;
      if (stmt->getRetValue()->isLValue() || stmt->getRetValue()->isXValue()) {
        assert(rv.isReference);
        val = rv.val;
      } else {
        val = rv.getValue(builder);
      }

      auto postTy = returnVal.getType().cast<MemRefType>().getElementType();
      if (auto prevTy = val.getType().dyn_cast<mlir::IntegerType>()) {
        auto ipostTy = postTy.cast<mlir::IntegerType>();
        if (prevTy != ipostTy) {
          val = builder.create<arith::TruncIOp>(loc, val, ipostTy);
        }
      } else if (val.getType().isa<MemRefType>() &&
                 postTy.isa<LLVM::LLVMPointerType>())
        val = builder.create<polygeist::Memref2PointerOp>(loc, postTy, val);
      else if (val.getType().isa<LLVM::LLVMPointerType>() &&
               postTy.isa<MemRefType>())
        val = builder.create<polygeist::Pointer2MemrefOp>(loc, postTy, val);
      builder.create<mlir::memref::StoreOp>(loc, val, returnVal);
    }
  }

  assert(loops.size() && "must be non-empty");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  for (auto l : loops) {
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.keepRunning);
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.noBreak);
  }

  return nullptr;
}
