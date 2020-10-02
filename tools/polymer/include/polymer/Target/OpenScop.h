//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TARGET_OPENSCOP_H
#define POLYMER_TARGET_OPENSCOP_H

namespace polymer {

void registerToOpenScopTranslation();
void registerFromOpenScopTranslation();

} // namespace polymer

#endif
