//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "polygeist/Ops.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining for Polygeist
/// dialect operations.
struct PolygeistInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All Polygeist dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "polygeist/PolygeistOps.cpp.inc"
      >();
  addInterfaces<PolygeistInlinerInterface>();
}

#include "polygeist/PolygeistOpsDialect.cpp.inc"
