//===- ArgumentList.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_CGEIST_ARGUMENTLIST_H
#define MLIR_TOOLS_CGEIST_ARGUMENTLIST_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

namespace mlirclang {
/// Class to pass options to a compilation tool.
class ArgumentList {
public:
  /// Add argument.
  ///
  /// The element stored will not be owned by this.
  void push_back(llvm::StringRef Arg) { Args.push_back(Arg.data()); }

  /// Add argument and ensure it will be valid before this passer's destruction.
  ///
  /// The element stored will be owned by this.
  template <typename... ArgTy> void emplace_back(ArgTy &&...Args) {
    // Store as a string
    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);
    (Stream << ... << Args);
    Storage.push_back(Stream.str());
    push_back(Storage.back());
  }

  /// Return the underling argument list.
  ///
  /// The return value of this operation could be invalidated by subsequent
  /// calls to push_back() or emplace_back().
  llvm::ArrayRef<const char *> getArguments() const { return Args; }

private:
  /// Helper storage.
  llvm::SmallVector<std::string> Storage;
  /// List of arguments
  llvm::SmallVector<const char *> Args;
};
} // end namespace mlirclang

#endif
