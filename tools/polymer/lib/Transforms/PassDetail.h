//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//

#ifndef POLYMER_TRANSFORMS_PASSDETAIL_H_
#define POLYMER_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "polymer/Transforms/Passes.h"

namespace polymer {
#define GEN_PASS_CLASSES
#include "polymer/Transforms/Passes.h.inc"
} // namespace polymer

#endif
