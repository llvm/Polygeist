//===- HLSPragmaParse.cc - Preprocessor stage of #pragma HLS --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Turns `#pragma HLS ...` (any case, `_Pragma`, or produced by macros) into
// PragmaRecords. Only syntax and option types are checked here; placement,
// name resolution and cross-option semantics are HLSPragmaSema.cc's job.
//
//===----------------------------------------------------------------------===//

#include "HLSPragmaInternal.h"

#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include <algorithm>

using namespace clang;
using namespace cgeist::hls;
using llvm::StringRef;

namespace {

/// Tokens of the option list, grouped into "words".
struct RawToken {
  Token tok;
  std::string spelling;
};

bool isIdentifierLike(const Token &T) {
  return T.getIdentifierInfo() != nullptr;
}

bool isPunct(const RawToken &T, StringRef s) {
  return !isIdentifierLike(T.tok) && T.tok.isNot(tok::numeric_constant) &&
         T.tok.isNot(tok::string_literal) && T.spelling == s;
}

/// Tokens that glue the value to the token before/after them even across
/// whitespace (`ports=a, b`, `foo< int >`, `N * 2`).
bool joinsNext(const RawToken &T) {
  return llvm::is_contained(
      std::initializer_list<StringRef>{",", ".", "::", "<", "(", "+", "-",
                                       "*", "/", "%", "<<", ">>", "->", "&",
                                       "|", "^", "~"},
      T.spelling);
}
bool joinsPrevious(const RawToken &T) {
  return llvm::is_contained(
      std::initializer_list<StringRef>{",", ".", "::", ">", ")", "+", "-",
                                       "*", "/", "%", "<<", ">>", "->", "&",
                                       "|", "^"},
      T.spelling);
}

} // namespace

/// `this` or `*this` as a pragma variable.
bool cgeist::hls::isThisReference(const std::vector<ValueToken> &v) {
  return (v.size() == 1 && v[0].text == "this") ||
         (v.size() == 2 && v[0].text == "*" && v[1].text == "this");
}

namespace {

ValueToken::Kind classify(const Token &T) {
  if (T.is(tok::numeric_constant))
    return ValueToken::Number;
  if (tok::isStringLiteral(T.getKind()))
    return ValueToken::String;
  if (isIdentifierLike(T))
    return ValueToken::Identifier;
  return ValueToken::Punct;
}

class PragmaParser {
public:
  PragmaParser(Preprocessor &PP, PragmaRecord &P, std::vector<RawToken> toks)
      : PP(PP), S(State::get()), P(P), toks(std::move(toks)) {}

  void parse();

private:
  Preprocessor &PP;
  State &S;
  PragmaRecord &P;
  std::vector<RawToken> toks;
  size_t pos = 0;

  bool atEnd() const { return pos >= toks.size(); }
  const RawToken &peek(size_t n = 0) const { return toks[pos + n]; }

  void fail() { P.valid = false; }
  std::vector<ValueToken> parseValue();
  void checkOption(ParsedOption &O);
};

void PragmaParser::parse() {
  std::string display = pragmaDisplayName(P);
  while (!atEnd()) {
    const RawToken &nameTok = peek();
    if (!isIdentifierLike(nameTok.tok)) {
      // Vitis 2025.1 warns ([HLS 207-5571] "unexpected pragma argument '2',
      // expects identifier") and drops the pragma for any token that is not
      // an option name (the open-source front end's parseIdentifierLoc,
      // ParseXlxPragma.cpp, gives the same warning for every non-identifier
      // token): hlsyn-vitis aes has `#pragma HLS pipeline` and `#pragma HLS
      // unroll 2` on loop L4, which Vitis pipelines (module
      // aes256_encrypt_ecb_Pipeline_L4), not unrolls; `#pragma HLS pipeline
      // =2` warns the same way and the loop is only pipelined automatically
      // ([HLS 214-376], Target II = NA; kernel w8bfix_peq, VITIS_NOTES N32).
      // cgeist rejected the `=` form with an error.
      S.warning(nameTok.tok.getLocation(),
                "unexpected argument '%1' in '%0', expected an option name; "
                "the pragma is ignored, as Vitis ignores it")
          << display << nameTok.spelling;
      fail();
      return;
    }
    ParsedOption O;
    O.spelling = nameTok.spelling;
    O.name = StringRef(nameTok.spelling).lower();
    O.loc = nameTok.tok.getLocation();
    ++pos;
    if (!atEnd() && isPunct(peek(), "=")) {
      ++pos;
      O.hasValue = true;
      if (atEnd()) {
        S.error(O.loc, "expected a value after '%0=' in '%1'")
            << O.spelling << display;
        fail();
        break;
      }
      O.value = parseValue();
    }
    checkOption(O);
    P.options.push_back(std::move(O));
  }

  if (P.isUnknown() || !P.valid)
    return;
  for (const OptionSpec &spec : P.spec->options) {
    if (spec.required && !P.get(spec.name)) {
      S.error(P.loc, "'%0' requires option '%1'")
          << display << spec.displayName();
      fail();
    }
  }
}

std::vector<ValueToken> PragmaParser::parseValue() {
  std::vector<ValueToken> value;
  int depth = 0;
  auto take = [&]() {
    const RawToken &T = toks[pos++];
    if (T.spelling == "(" || T.spelling == "<")
      ++depth;
    else if (T.spelling == ")" || T.spelling == ">")
      --depth;
    else if (T.spelling == ">>")
      depth -= 2;
    value.push_back({classify(T.tok), T.spelling, T.tok.getLocation()});
  };
  take(); // the first token always belongs to the value (`key= value`)
  // Whitespace separates options, except inside brackets and around
  // operators and separators.
  while (!atEnd()) {
    const RawToken &next = peek();
    const RawToken &last = toks[pos - 1];
    if (isPunct(next, "="))
      break;
    bool glued = depth > 0 || !next.tok.hasLeadingSpace() ||
                 joinsNext(last) || joinsPrevious(next);
    if (!glued)
      break;
    take();
  }
  return value;
}

/// Evaluates a flag value (`true`, `false`, `1`, `0`); nullopt if invalid.
std::optional<bool> flagValue(const ParsedOption &O) {
  if (!O.hasValue)
    return true;
  if (O.value.size() != 1)
    return std::nullopt;
  std::string v = StringRef(O.value[0].text).lower();
  if (v == "true" || v == "1" || v == "on" || v == "yes")
    return true;
  if (v == "false" || v == "0" || v == "off" || v == "no")
    return false;
  return std::nullopt;
}

bool isIntegerExpressionToken(const ValueToken &T) {
  if (T.kind == ValueToken::Number || T.kind == ValueToken::Identifier)
    return true;
  return llvm::is_contained(
      std::initializer_list<StringRef>{"+", "-", "*", "/", "%", "(", ")",
                                       "<<", ">>", "&", "|", "^", "~", "::", ","},
      T.text);
}

/// Whether a numeric literal is an integer (decimal, octal, hex, binary).
bool isIntegerLiteral(StringRef text) {
  text = text.rtrim("uUlLzZ");
  if (text.starts_with_insensitive("0x"))
    return text.size() > 2 && llvm::all_of(text.drop_front(2), llvm::isHexDigit);
  if (text.starts_with_insensitive("0b"))
    return text.size() > 2 &&
           llvm::all_of(text.drop_front(2), [](char c) { return c == '0' || c == '1'; });
  return !text.empty() && llvm::all_of(text, llvm::isDigit);
}

bool isIntegerOperand(const ValueToken &T) {
  return isIntegerExpressionToken(T) &&
         (T.kind != ValueToken::Number || isIntegerLiteral(T.text));
}

void PragmaParser::checkOption(ParsedOption &O) {
  std::string display = pragmaDisplayName(P);
  if (P.isUnknown())
    return; // preserved verbatim, nothing to check

  if (O.name == "off" && P.spec->hasOff)
    O.spec = P.spec->findOption("off");
  else
    O.spec = P.spec->findOption(O.name);

  // Positional enum value, e.g. `array_partition cyclic`.
  if (!O.spec && !O.hasValue) {
    if (const OptionSpec *pos = P.spec->findPositional(O.name)) {
      O.spec = pos;
      O.positional = true;
      O.hasValue = true;
      O.value = {{ValueToken::Identifier, O.spelling, O.loc}};
      O.name = pos->name.str();
    } else if (P.spec->name == "dependence" &&
               (O.name == "true" || O.name == "false")) {
      // Vivado spelling: `#pragma HLS dependence variable=a inter false`.
      O.spec = P.spec->findOption("dependent");
      O.positional = true;
      O.hasValue = true;
      O.value = {{ValueToken::Identifier, O.name, O.loc}};
      O.name = "dependent";
    }
  }

  if (!O.spec) {
    // Vitis 2025.1 drops an INTERFACE pragma with an option it does not know
    // as a whole ([HLS 207-5570] "unexpected pragma parameter"; the open
    // front end's XlxPragmaArgParser::parse returns false): kernel
    // w9r14p_mx_bw (`m_axi ... bus_bitwidth=512`) gets an ap_memory port
    // and no adapter, and the pragma's other options are never checked
    // (its max_read_burst_length=512 is not rejected, as it is without
    // bus_bitwidth: w9r14p_mx_nobw, [HLS 214-326]); class_ece8893
    // max_beaulieu lab3 (VITIS_NOTES N108). Every pragma goes through the
    // same parser in the open front end, and Vitis drops the others the
    // same way (CIRCT W9r14 review kernels w9r14rv_mx_ap: array_partition
    // with an unknown option, the array stays one memory, 6; w9r14rv_mx_bs:
    // bind_storage impl=bram with an unknown option, 0; max_beaulieu's
    // `unroll II=1`, 207-5570).
    S.warning(O.loc, "unknown option '%0' in '%1'; the pragma is ignored, "
                     "as Vitis ignores it")
        << O.spelling << display;
    fail();
    return;
  }

  // Duplicate options.
  for (const ParsedOption &prev : P.options) {
    if (prev.spec == O.spec) {
      S.error(O.loc, "option '%0' is specified more than once in '%1'")
          << O.spec->displayName() << display;
      S.note(prev.loc, "previous occurrence is here");
      fail();
      return;
    }
  }

  const OptionSpec &spec = *O.spec;
  auto badValue = [&](StringRef expected) {
    S.error(O.value.empty() ? O.loc : O.value.front().loc,
            "invalid value '%0' for option '%1' in '%2': expected %3")
        << O.valueText() << spec.displayName() << display << expected;
    fail();
  };

  if (spec.type != OptType::Flag && !O.hasValue) {
    S.error(O.loc, "option '%0' in '%1' requires a value")
        << spec.displayName() << display;
    fail();
    return;
  }

  switch (spec.type) {
  case OptType::Flag:
    if (!flagValue(O))
      badValue("'true' or 'false'");
    return;
  case OptType::Enum:
  case OptType::IntOrEnum: {
    if (O.value.size() == 1 && O.value[0].kind == ValueToken::Identifier) {
      std::string lower = StringRef(O.value[0].text).lower();
      if (llvm::is_contained(spec.values, StringRef(lower)))
        return;
      if (spec.type == OptType::Enum) {
        std::string list;
        for (StringRef v : spec.values)
          list += (list.empty() ? "" : ", ") + v.str();
        badValue("one of: " + list);
        return;
      }
    }
    if (spec.type == OptType::IntOrEnum &&
        llvm::all_of(O.value, isIntegerOperand))
      return;
    std::string list;
    for (StringRef v : spec.values)
      list += (list.empty() ? "" : ", ") + v.str();
    badValue(spec.type == OptType::Enum ? "one of: " + list
                                        : "an integer or one of: " + list);
    return;
  }
  case OptType::Int:
    if (!llvm::all_of(O.value, isIntegerOperand))
      badValue("an integer constant");
    return;
  case OptType::Number:
    if (!llvm::all_of(O.value, isIntegerExpressionToken))
      badValue("a number");
    return;
  case OptType::Variable: {
    // ['*'] 'this'
    // | ident ('::' ident)* ( ('.' | '->') ident | '[' expr ']' )*
    // `this` names the enclosing object (Vitis hls_vector.h:110,
    // hls_stream_39.h:162); the qualified name is a namespace-scope variable
    // (Vitis etc/hls_hotbm_apfixed.h:91), as accepted by hls-llvm-project
    // ee930469 ParseXlxPragma.cpp (variable= is parsed as an expression).
    // Both forms are accepted only under --xlx-headers, so that the output
    // without it stays as before (`variable=this` is an error there).
    bool xlx = cgeist::hls::xlxHeaders();
    if (xlx && isThisReference(O.value))
      return;
    // `variable=&s`: Vitis 2025.1 drops the `&` with [HLS 207-5544] "extra
    // token before variable expression is ignored" (kernel
    // w9r22i_c6_addr_var; Vitis_Libraries vision xf_cvt_color_1.hpp
    // `variable=&uvStream`).
    if (O.value.size() > 1 && O.value[0].text == "&" &&
        O.value[1].kind == ValueToken::Identifier) {
      S.warning(O.value[0].loc, "extra token before variable expression is "
                                "ignored in '%0', as Vitis ignores it")
          << display;
      O.value.erase(O.value.begin());
    }
    const std::vector<ValueToken> &v = O.value;
    bool ok = !v.empty() && v[0].kind == ValueToken::Identifier;
    size_t i = 1;
    while (ok && xlx && i + 1 < v.size() && v[i].text == "::" &&
           v[i + 1].kind == ValueToken::Identifier)
      i += 2;
    while (ok && i < v.size()) {
      if ((v[i].text == "." || v[i].text == "->") && i + 1 < v.size() &&
          v[i + 1].kind == ValueToken::Identifier) {
        i += 2;
      } else if (v[i].text == "[") {
        size_t close = i + 1;
        while (close < v.size() && v[close].text != "]")
          ++close;
        ok = close < v.size() && close > i + 1 &&
             std::all_of(v.begin() + i + 1, v.begin() + close,
                         isIntegerOperand);
        i = close + 1;
      } else {
        ok = false;
      }
    }
    if (!ok)
      badValue("a variable name");
    return;
  }
  case OptType::Name:
    return;
  case OptType::NameList:
  case OptType::IntList: {
    bool expectItem = true;
    for (const ValueToken &T : O.value) {
      if (T.text == ",") {
        if (expectItem)
          return badValue("a comma-separated list");
        expectItem = true;
      } else {
        if (spec.type == OptType::IntList && !isIntegerOperand(T))
          return badValue("a comma-separated list of integers");
        expectItem = false;
      }
    }
    if (expectItem)
      badValue("a comma-separated list");
    return;
  }
  }
}

/// Catch-all handler: registered without a name at the root namespace, so it
/// sees every pragma clang has no handler for. That is how `hls`, `Hls` etc.
/// are recognised (clang's namespace lookup is case-sensitive).
class HLSPragmaHandler : public PragmaHandler {
public:
  HLSPragmaHandler() : PragmaHandler(StringRef()) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &FirstTok) override {
    if (!FirstTok.getIdentifierInfo() ||
        !FirstTok.getIdentifierInfo()->getName().equals_insensitive("hls")) {
      // What clang does when no handler matches; the preprocessor discards
      // the rest of the line.
      PP.Diag(FirstTok, diag::warn_pragma_ignored);
      return;
    }

    // Lex (with macro expansion) up to the end of the pragma.
    std::vector<RawToken> toks;
    Token T;
    PP.Lex(T);
    while (T.isNot(tok::eod) && T.isNot(tok::eof)) {
      toks.push_back({T, PP.getSpelling(T)});
      PP.Lex(T);
    }

    State &S = State::get();
    if (!isEnabled())
      return; // consumed silently, as before --hls-annotate existed

    PragmaRecord P;
    P.id = S.pragmas.size();
    P.loc = Introducer.Loc;
    P.locString = S.locString(Introducer.Loc);

    // Normalized text, used for diagnostics and hls.unknown_pragmas.
    for (size_t i = 0; i < toks.size(); ++i) {
      if (i && toks[i].tok.hasLeadingSpace())
        P.text += ' ';
      P.text += toks[i].spelling;
    }

    P.report = S.report.size();
    S.report.push_back({P.locString, "#pragma HLS " + P.text, {}});

    // An empty `#pragma HLS` is accepted silently by Vitis 2025.1
    // (xf_max_suppression.hpp:203; no message in the harris logs): ignored.
    if (toks.empty()) {
      S.addOutcome(P.report, "ignored: empty pragma (as Vitis)");
      return;
    }
    if (!isIdentifierLike(toks[0].tok)) {
      S.error(toks.empty() ? FirstTok.getLocation()
                           : toks[0].tok.getLocation(),
              "expected a pragma name after '#pragma HLS'");
      S.addOutcome(P.report, "rejected: invalid syntax");
      return;
    }
    P.kindName = StringRef(toks[0].spelling).lower();
    P.spec = lookupPragmaSpec(P.kindName);
    if (P.isUnknown())
      S.warning(toks[0].tok.getLocation(),
                "unknown HLS pragma '%0'; it is preserved in "
                "'hls.unknown_pragmas'")
          << toks[0].spelling;

    toks.erase(toks.begin());
    PragmaParser(PP, P, std::move(toks)).parse();
    if (!P.valid)
      S.addOutcome(P.report, "rejected: invalid syntax");
    S.pragmas.push_back(std::move(P));
  }
};

} // namespace

std::string ParsedOption::valueText() const {
  std::string s;
  for (const ValueToken &T : value)
    s += T.text;
  return s;
}

const ParsedOption *PragmaRecord::get(StringRef name) const {
  for (const ParsedOption &O : options)
    if (O.spec && O.spec->name == name)
      return &O;
  return nullptr;
}

bool PragmaRecord::isOff() const {
  if (!spec || !spec->hasOff)
    return false;
  const ParsedOption *O = get("off");
  if (!O)
    return false;
  std::optional<bool> v = flagValue(*O);
  return v && *v;
}

std::string cgeist::hls::pragmaDisplayName(const PragmaRecord &P) {
  return "#pragma HLS " + P.kindName;
}

void cgeist::hls::registerPragmaHandler(Preprocessor &PP) {
  PP.AddPragmaHandler(new HLSPragmaHandler());
}
