//===- XlxMathCore.cc - hls:: math library calls as cores ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See XlxMathCore.h. The function names are those hls-est's math-core model
// knows (MathCores.cpp coreFunctions); hls_math.h spells them also with an
// `f` suffix (hls::expf, hls::sqrtf) and a `half_` prefix (hls::half_exp).
// The fixed-point overloads are the ones the stand-in headers mark
// (shim/exactclass/hls_math.h HLSEST_M1 / HLSEST_M2); `abs` and the rounding
// functions are computed in plain logic there and are not cores.
//
//===----------------------------------------------------------------------===//

#include "XlxMathCore.h"
#include "HLSPragma.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"

using namespace clang;

namespace cgeist {
namespace xlx {

namespace {

/// Cores of the float / double / half overloads (MathCores coreFunctions).
const llvm::StringSet<> &floatCores() {
  static const llvm::StringSet<> s = {
      "exp",   "exp2",  "exp10", "expm1", "log",   "log2",  "log10",
      "log1p", "logb",  "sqrt",  "rsqrt", "cbrt",  "recip", "sin",
      "cos",   "tan",   "sinpi", "cospi", "asin",  "acos",  "atan",
      "atan2", "sinh",  "cosh",  "tanh",  "asinh", "acosh", "atanh",
      "pow",   "erf",   "erfc",  "hypot", "fmod",  "remainder"};
  return s;
}

/// Cores of the fixed-point overloads (the stand-in headers' markers).
const llvm::StringSet<> &fixedCores() {
  static const llvm::StringSet<> s = {
      "sqrt",  "exp",   "exp2",  "expm1", "log",  "log2",  "log10", "log1p",
      "logb",  "tanh",  "sinh",  "cosh",  "atanh", "asinh", "acosh", "erf",
      "erfc",  "cbrt",  "rsqrt", "tan",   "sinpi", "cospi", "sin",  "cos",
      "atan",  "asin",  "acos",  "atan2", "pow"};
  return s;
}

/// W, I and signedness of an ap_fixed_base / ap_int_base class or a class
/// over one (ap_fixed<W, I>, ap_uint<W>).
bool apFormat(const CXXRecordDecl *rd, int64_t &w, int64_t &i, bool &s) {
  for (unsigned guard = 0; rd && guard < 8; ++guard) {
    if (auto *spec = dyn_cast<ClassTemplateSpecializationDecl>(rd)) {
      const TemplateArgumentList &a = spec->getTemplateArgs();
      auto num = [&](unsigned k, int64_t &v) {
        if (k >= a.size() || a[k].getKind() != TemplateArgument::Integral)
          return false;
        v = a[k].getAsIntegral().getExtValue();
        return true;
      };
      int64_t sv = 0;
      if (spec->getName() == "ap_fixed_base" && a.size() == 6)
        if (num(0, w) && num(1, i) && num(2, sv)) {
          s = sv != 0;
          return true;
        }
      if (spec->getName() == "ap_int_base" && a.size() == 2)
        if (num(0, w) && num(1, sv)) {
          i = w;
          s = sv != 0;
          return true;
        }
    }
    if (rd->getNumBases() != 1)
      return false;
    rd = rd->bases_begin()->getType()->getAsCXXRecordDecl();
  }
  return false;
}

bool declaredInHlsMath(const FunctionDecl *fd, ASTContext &ctx) {
  const FunctionDecl *pattern = fd->getTemplateInstantiationPattern();
  const FunctionDecl *d = pattern ? pattern : fd;
  const SourceManager &sm = ctx.getSourceManager();
  SourceLocation loc = sm.getExpansionLoc(d->getLocation());
  if (loc.isInvalid())
    return false;
  return llvm::sys::path::filename(sm.getFilename(loc)) == "hls_math.h";
}

} // namespace

std::optional<MathCore> mathCoreOf(const FunctionDecl *fd, ASTContext &ctx) {
  if (!cgeist::hls::xlxHeaders() || !fd || !fd->getIdentifier() ||
      fd->getNumParams() == 0 || isa<CXXMethodDecl>(fd))
    return std::nullopt;
  auto *ns = dyn_cast<NamespaceDecl>(fd->getDeclContext()->getRedeclContext());
  if (!ns || ns->getName() != "hls" ||
      !ns->getDeclContext()->getRedeclContext()->isTranslationUnit())
    return std::nullopt;
  if (!declaredInHlsMath(fd, ctx))
    return std::nullopt;

  QualType arg = fd->getParamDecl(0)->getType().getNonReferenceType();
  arg = arg.getUnqualifiedType();
  llvm::StringRef name = fd->getName();
  if (const CXXRecordDecl *rd = arg->getAsCXXRecordDecl()) {
    int64_t w = 0, i = 0;
    bool s = true;
    if (!fixedCores().contains(name) || !apFormat(rd, w, i, s))
      return std::nullopt;
    return MathCore{name.str(), (s ? "ap_fixed<" : "ap_ufixed<") +
                                    std::to_string(w) + "," +
                                    std::to_string(i) + ">"};
  }
  if (!arg->isRealFloatingType())
    return std::nullopt;
  std::string fn = name.str();
  if (llvm::StringRef(fn).starts_with("half_"))
    fn = fn.substr(5);
  if (!floatCores().contains(fn) && fn.size() > 1 && fn.back() == 'f' &&
      floatCores().contains(llvm::StringRef(fn).drop_back()))
    fn.pop_back();
  if (!floatCores().contains(fn))
    return std::nullopt;
  return MathCore{fn, ""};
}

} // namespace xlx
} // namespace cgeist
