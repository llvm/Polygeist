#include "pragmaHandler.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
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
  /// TODO: Handle assertions properly.
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &PragmaTok) {
    Token Tok;
    PP.Lex(Tok); // lparen
    assert(Tok.is(tok::l_paren) && "lower_to should start with '('.");

    Token FuncIdTok; // function identifier
    PP.Lex(FuncIdTok);
    assert(FuncIdTok.is(tok::identifier) &&
           "The first argument of lower_to should be an identifier.");

    llvm::StringRef FuncId = FuncIdTok.getIdentifierInfo()->getName();

    PP.Lex(Tok); // comma
    assert(Tok.is(tok::comma) && "The first and second argument of lower_to "
                                 "should be separated by a comma.");

    // Parse the string literal argument, which is the MLIR function symbol.
    SmallVector<Token, 1> SymbolToks;
    Token SymbolTok;
    PP.Lex(SymbolTok);
    assert(SymbolTok.is(tok::string_literal) &&
           "The second argument of lower_to should be a string literal.");
    SymbolToks.push_back(SymbolTok);
    StringRef SymbolName = StringLiteralParser(SymbolToks, PP).GetString();
    PP.Lex(Tok); // rparen
    assert(Tok.is(tok::r_paren) && "lower_to should end with '('.");

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
