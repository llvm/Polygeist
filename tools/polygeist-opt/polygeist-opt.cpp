//===- polygeist-opt.cpp - The polygeist-opt driver -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'polygeist-opt' tool, which is the polygeist analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  registry.insert<mlir::polygeist::PolygeistDialect>();

  mlir::registerpolygeistPasses();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
 
  auto f = [](MLIRContext &context) { 
	  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
	  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
	  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);
	  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
		  context);
	  LLVM::LLVMPointerType::attachInterface<
		  PtrElementModel<LLVM::LLVMPointerType>>(context);
	  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
		  context);
  };

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Polygeist modular optimizer driver", registry,
      /*preloadDialectsInContext=*/false));
}
