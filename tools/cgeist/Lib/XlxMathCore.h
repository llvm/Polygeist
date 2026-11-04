//===- XlxMathCore.h - hls:: math library calls as cores ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// --xlx-headers: a call of an hls:: math function of the real Vitis HLS
// hls_math.h (hls::exp, hls::tanh, hls::sqrt, ... of ap_fixed, float, double
// or half) is one core of the Vitis math library, whose BRAM hls-est prices
// per instance from its cost table (hls-est MathCores.cpp, VITIS_NOTES N16,
// tools/hls-est-validate/math_costs.json) - the model the stand-in headers
// feed through their __hlsest_math_<fn> markers. cgeist marks such a call
// with `hls.math_core = {function = "<fn>", type = "<type>"}` (type only for
// the fixed-point overloads: `ap_fixed<W,I>` / `ap_ufixed<W,I>` of the
// argument; the float / double / half type is the call's own operand type)
// and keeps the callee as a function of its own (`hls.math_core` on the
// func.func: never inlined), so that the library's code and tables are not
// mistaken for the design's while the MLIR still computes the headers'
// values. hls-est drops the body of such a callee.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_XLXMATHCORE_H
#define CGEIST_XLXMATHCORE_H

#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace clang {
class ASTContext;
class FunctionDecl;
} // namespace clang

namespace cgeist {
namespace xlx {

/// The attribute on the call and on the kept callee.
constexpr llvm::StringLiteral kMathCoreAttr = "hls.math_core";

struct MathCore {
  std::string function; // exp, tanh, sqrt, ... (MathCores' names)
  std::string type;     // ap_fixed<W,I> / ap_ufixed<W,I>, or empty
};

/// If `fd` (the direct callee of a call) is a math function of the Vitis
/// hls_math.h (declared in namespace hls in a file named hls_math.h), named
/// like a core of the math library: the core. None otherwise, and always
/// without --xlx-headers.
std::optional<MathCore> mathCoreOf(const clang::FunctionDecl *fd,
                                   clang::ASTContext &ctx);

} // namespace xlx
} // namespace cgeist

#endif // CGEIST_XLXMATHCORE_H
