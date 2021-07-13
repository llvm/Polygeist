#include "pragmaHandler.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

namespace {

/// Handles the #pragma lower_to(<identifier>, "<mlir function target>")
/// directive.
class PragmaLowerToHandler : public PragmaHandler {
  LowerToInfo &Info;

public:
  PragmaLowerToHandler(LowerToInfo &Info)
      : PragmaHandler("lower_to"), Info(Info) {}

  /// The pragma handler will extract the single argument to the lower_to(...)
  /// pragma definition, which is the target MLIR function symbol, and relate
  /// the function decl that lower_to is attached to with that MLIR function
  /// symbol in the class-referenced dictionary.
  ///
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &PragmaTok) {
    Token Tok;
    PP.Lex(Tok); // lparen
    if (Tok.isNot(tok::l_paren)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen)
          << "lower_to";
      return;
    }

    Token PrevTok = Tok;
    llvm::StringRef FuncId = llvm::StringRef();
    llvm::StringRef SymbolName = llvm::StringRef();
    while (Tok.isNot(tok::eod)) {
      Token CurrentTok;
      PP.Lex(CurrentTok);

      // rparen.
      if (PrevTok.is(tok::string_literal)) {
        if (CurrentTok.isNot(tok::r_paren)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_rparen)
              << "lower_to";
          return;
        } else {
          break;
        }
      }

      // function identifier.
      if (PrevTok.is(tok::l_paren)) {
        if (CurrentTok.isNot(tok::identifier)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
              << "lower_to";
          return;
        } else {
          FuncId = CurrentTok.getIdentifierInfo()->getName();
        }
      }

      // comma.
      if (PrevTok.is(tok::identifier)) {
        if (CurrentTok.isNot(tok::comma)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_punc)
              << "lower_to";
          return;
        }
      }

      // string literal, which is the MLIR function symbol.
      if (PrevTok.is(tok::comma)) {
        if (CurrentTok.isNot(tok::string_literal)) {
          PP.Diag(CurrentTok.getLocation(),
                  diag::warn_pragma_expected_section_name)
              << "lower to";
          return;
        } else {
          SmallVector<Token, 1> SymbolToks;
          SymbolToks.push_back(CurrentTok);
          SymbolName = StringLiteralParser(SymbolToks, PP).GetString();
        }
      }

      PrevTok = CurrentTok;
    }

    // Link SymbolName with the function.
    auto result = Info.SymbolTable.try_emplace(FuncId, SymbolName);
    assert(result.second &&
           "Shouldn't define lower_to over the same func id more than once.");
  }

private:
};

struct PragmaScopHandler : public PragmaHandler {
  ScopLocList &scops;

  PragmaScopHandler(ScopLocList &scops) : PragmaHandler("scop"), scops(scops) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &scopTok) override {
    auto &SM = PP.getSourceManager();
    auto loc = scopTok.getLocation();
    scops.addStart(SM, loc);
  }
};

struct PragmaEndScopHandler : public PragmaHandler {
  ScopLocList &scops;

  PragmaEndScopHandler(ScopLocList &scops)
      : PragmaHandler("endscop"), scops(scops) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer introducer,
                    Token &endScopTok) override {
    auto &SM = PP.getSourceManager();
    auto loc = endScopTok.getLocation();
    scops.addEnd(SM, loc);
  }
};

} // namespace

void addPragmaLowerToHandlers(Preprocessor &PP, LowerToInfo &LTInfo) {
  PP.AddPragmaHandler(new PragmaLowerToHandler(LTInfo));
}

void addPragmaScopHandlers(Preprocessor &PP, ScopLocList &scopLocList) {
  PP.AddPragmaHandler(new PragmaScopHandler(scopLocList));
}

void addPragmaEndScopHandlers(Preprocessor &PP, ScopLocList &scopLocList) {
  PP.AddPragmaHandler(new PragmaEndScopHandler(scopLocList));
}
