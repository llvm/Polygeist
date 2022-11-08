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
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>

namespace mlirclang {
/// Class to pass options to a compilation tool.
class ArgumentList {
private:
  /// Helper storage.
  llvm::SmallVector<llvm::SmallString<0>> Storage;
  /// List of arguments
  llvm::SmallVector<const char *> Args;

public:
  /// Add argument.
  ///
  /// The element stored will not be owned by this.
  void push_back(const char *Arg) { Args.push_back(Arg); }

  /// Add argument and ensure it will be valid before this passer's destruction.
  ///
  /// The element stored will be owned by this.
  template <typename... ArgTy> void emplace_back(ArgTy &&...Args) {
    // Store as a string
    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);
    (Stream << ... << Args);
    emplace_back(llvm::StringRef(Stream.str()));
  }

  void emplace_back(llvm::StringRef &&Arg) {
    push_back(Storage.emplace_back(Arg).c_str());
  }

  /// Return the underling argument list.
  ///
  /// The return value of this operation could be invalidated by subsequent
  /// calls to push_back() or emplace_back().
  llvm::ArrayRef<const char *> getArguments() const { return Args; }
};
} // end namespace mlirclang

#endif
