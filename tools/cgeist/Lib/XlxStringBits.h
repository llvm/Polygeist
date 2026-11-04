//===- XlxStringBits.h - Fold ap_int / ap_fixed string constructors -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// --xlx-headers: compile-time values of the Vitis HLS headers' string
// constructors (`ap_ufixed<25,11> t[] = {"0x1.a614p0", ...}`, the lookup
// tables of the fixed-point math functions). Under __SYNTHESIS__ these call
// `_ssdm_string2bits(&t, s, radix, I, S, Q, O, N, C99)`
// (etc/ap_fixed_base.h:458-471, etc/ap_int_base.h:398-410), which the
// closed Vitis front end evaluates at compile time ("XFORM will do compile
// time processing of the string"). See XlxStringBits.cc.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_XLXSTRINGBITS_H
#define CGEIST_XLXSTRINGBITS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace clang {
class ASTContext;
class Expr;
} // namespace clang

namespace cgeist {
namespace xlx {

/// The target format of _ssdm_string2bits: W bits, I integer bits, signed or
/// not, and the quantisation / overflow modes by their enumerator names
/// (AP_TRN, AP_RND_CONV, ..., AP_WRAP, AP_SAT_SYM, ...), N saturation bits.
struct FixedFormat {
  unsigned W = 0;
  int64_t I = 0;
  bool S = false;
  llvm::StringRef Q = "AP_TRN";
  llvm::StringRef O = "AP_WRAP";
  int64_t N = 0;
};

/// The W-bit pattern of the string `str` (a C-simulation style number:
/// optional sign, 0x / 0b / 0o prefix, digits with an optional point, a
/// decimal `e` or binary `p` exponent) in the format `fmt`, read in radix
/// `radix` when there is no prefix (0 or 10: decimal). None when the string is
/// not a number of that form or the format is not supported.
std::optional<llvm::APInt> stringToFixedBits(llvm::StringRef str,
                                             unsigned radix,
                                             const FixedFormat &fmt);

/// If `E` constructs (through temporaries, elidable copies and casts) a
/// scalar-wrapper object with a constructor that passes its string argument
/// to _ssdm_string2bits, and that argument is a string literal: the object's
/// bit pattern. None otherwise.
std::optional<llvm::APInt> foldStringConstruct(const clang::Expr *E,
                                               clang::ASTContext &Ctx);

/// An ap_int / ap_uint / ap_fixed / ap_ufixed default construction (a
/// CXXConstructExpr without arguments of one of those classes): its value in
/// static storage, 0 at the type's width (the object is zero-initialised
/// before the constructor runs, and the headers' default constructors set
/// nothing). None otherwise.
std::optional<llvm::APInt> foldDefaultConstruct(const clang::Expr *E,
                                                clang::ASTContext &Ctx);

/// If `E` constructs (through temporaries, elidable copies and casts) an
/// ap_int / ap_uint / ap_fixed / ap_ufixed object (a class over
/// ap_int_base<W, S> or ap_fixed_base<W, I, S, Q, O, N>) from one integer,
/// bool or (for the fixed-point classes) floating-point constant: the object's
/// bit pattern, the constant's exact value quantised and wrapped / saturated
/// by the class's modes as the headers' numeric constructors do
/// (etc/ap_fixed_base.h:547-640 for double, the integer constructors through
/// ap_int_base). None otherwise, and for N > 0 or AP_WRAP_SM.
std::optional<llvm::APInt> foldNumericConstruct(const clang::Expr *E,
                                                clang::ASTContext &Ctx);

} // namespace xlx
} // namespace cgeist

#endif // CGEIST_XLXSTRINGBITS_H
