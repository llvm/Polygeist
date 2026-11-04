//===- HLSPragmaInternal.h - HLS pragma data model --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data model shared by the parser (HLSPragmaParse.cc), the association and
// validation stage (HLSPragmaSema.cc) and materialization
// (HLSPragmaMaterialize.cc). Not part of the public interface.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_LIB_HLSPRAGMAINTERNAL_H
#define CGEIST_LIB_HLSPRAGMAINTERNAL_H

#include "HLSPragma.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clang {
class SourceManager;
} // namespace clang

namespace cgeist {
namespace hls {

//===----------------------------------------------------------------------===//
// Pragma specifications
//===----------------------------------------------------------------------===//

/// Where a pragma of a given kind attaches.
enum class Placement {
  Loop,           ///< innermost enclosing loop; error outside loops
  LoopOrFunction, ///< innermost enclosing loop, else the function
  Function,       ///< enclosing function
  Variable,       ///< the storage named by the `variable` option
  Port,           ///< the parameter named by `port` (or `return`)
};

/// Value type of an option.
enum class OptType {
  Flag,     ///< boolean; bare `opt`, or `opt=true|false|1|0`
  Int,      ///< integer constant expression (macros, constexpr, enums)
  Number,   ///< integer or floating point literal
  Enum,     ///< one identifier out of a fixed, case-insensitive set
  Variable, ///< C identifier, optionally a member path `a.b.c`
  Name,     ///< free-form name (e.g. `foo<int>`, a bundle name)
  NameList, ///< comma-separated names
  IntList,  ///< comma-separated integer constant expressions
  IntOrEnum ///< integer constant expression, or one of an enum set
};

struct OptionSpec {
  llvm::StringRef name;
  OptType type;
  std::vector<llvm::StringRef> values; ///< Enum/IntOrEnum values
  bool required = false;
  /// Enum values may be written bare (`#pragma HLS array_partition cyclic`).
  bool positional = false;
  std::optional<int64_t> min;
  /// Key in the emitted dictionary (defaults to `name`).
  llvm::StringRef attrKey;
  /// Default emitted when the option is omitted.
  enum class DefaultKind { None, Bool, Int, String } defaultKind =
      DefaultKind::None;
  bool defaultBool = false;
  int64_t defaultInt = 0;
  llvm::StringRef defaultString;
  /// Emit enum values upper-case (`direction=raw` -> "RAW").
  bool upperCase = false;

  llvm::StringRef key() const { return attrKey.empty() ? name : attrKey; }
  /// Spelling used in diagnostics (`II`, not `ii`).
  llvm::StringRef displayName() const {
    return key().equals_insensitive(name) ? key() : name;
  }
};

enum class Multiplicity {
  Single,      ///< one per target
  PerVariable, ///< one per (target, variable[, dim])
  Many,        ///< any number; only exact duplicates are diagnosed
};

struct PragmaSpec {
  llvm::StringRef name; ///< canonical lower-case name
  Placement placement;
  std::vector<OptionSpec> options;
  bool hasOff = false; ///< supports `off` -> `hls.<name>_off`
  Multiplicity multiplicity = Multiplicity::Single;

  const OptionSpec *findOption(llvm::StringRef lowerName) const;
  /// Returns the option whose enum values contain `lowerValue` and that may
  /// be written positionally.
  const OptionSpec *findPositional(llvm::StringRef lowerValue) const;
};

/// All known pragmas. Lookup is by lower-case name.
const PragmaSpec *lookupPragmaSpec(llvm::StringRef lowerName);
llvm::ArrayRef<PragmaSpec> allPragmaSpecs();

//===----------------------------------------------------------------------===//
// Parsed pragmas
//===----------------------------------------------------------------------===//

/// One token of an option value, after macro expansion.
struct ValueToken {
  enum Kind { Number, Identifier, String, Punct } kind;
  std::string text;
  clang::SourceLocation loc;
};

/// `this` or `*this` written as a pragma variable (Vitis header pragmas such
/// as `#pragma HLS aggregate variable=this`).
bool isThisReference(const std::vector<ValueToken> &v);

struct ParsedOption {
  std::string name; ///< canonical lower-case option name
  std::string spelling;
  clang::SourceLocation loc;
  const OptionSpec *spec = nullptr;
  /// Positional enum value written without `name=`.
  bool positional = false;
  bool hasValue = false;
  std::vector<ValueToken> value;

  std::string valueText() const;
};

struct PragmaRecord {
  unsigned id = 0;
  const PragmaSpec *spec = nullptr; ///< null for unknown pragmas
  std::string kindName;             ///< lower-case pragma name as written
  clang::SourceLocation loc;        ///< location of `#pragma` / `_Pragma`
  std::string text;                 ///< normalized text after `HLS`
  std::string locString;            ///< "file:line:col" for late diagnostics
  std::vector<ParsedOption> options;
  bool valid = true;      ///< false after a syntax error
  bool associated = false; ///< bound to a function that is emitted
  int report = -1;         ///< index into State::report

  bool isUnknown() const { return spec == nullptr; }
  const ParsedOption *get(llvm::StringRef name) const;
  bool isOff() const;
};

//===----------------------------------------------------------------------===//
// Location tags and per-process state
//===----------------------------------------------------------------------===//

/// What kind of op a tag may be materialized on.
enum class TagClass { Loop, Storage, Operation };

/// One target that is identified through a location tag.
struct TagRecord {
  unsigned id = 0;
  TagClass cls = TagClass::Loop;
  /// Attributes to add: name -> ArrayAttr of entries, or UnitAttr.
  llvm::SmallVector<std::pair<std::string, mlir::Attribute>, 2> attrs;
  /// For TagClass::Operation: accepted op names (e.g. "arith.muli").
  llvm::SmallVector<std::string, 4> opNames;
  /// For TagClass::Operation: required float width (0 = integer op / any).
  unsigned floatWidth = 0;
  std::string pragmaName; ///< for diagnostics
  std::string what;       ///< e.g. "loop", "variable 'buf'"
  std::string locString;
  std::string funcName; ///< symbol of the function containing the target
  llvm::SmallVector<int, 2> reports; ///< State::report entries it serves
  bool tagged = false;       ///< some op received the tag during codegen
  bool materialized = false; ///< some op received the attributes
};

/// Pending attributes for a global variable.
struct GlobalTarget {
  llvm::SmallVector<std::pair<std::string, mlir::Attribute>, 2> attrs;
  std::string pragmaName;
  std::string name;
  clang::SourceLocation loc;
  mlir::Operation *op = nullptr;
  llvm::SmallVector<int, 2> reports;
};

/// What finally happened to one pragma (--hls-pragma-report).
struct ReportEntry {
  std::string loc;
  std::string pragma; ///< "#pragma HLS <text>"
  std::vector<std::string> outcomes;
};

class State {
public:
  static State &get();

  // Per translation unit.
  clang::Preprocessor *PP = nullptr;
  mlir::MLIRContext *mlirCtx = nullptr;
  std::vector<PragmaRecord> pragmas;
  /// Pragma ids already associated (prevents duplicate diagnostics when the
  /// same body is visited twice).
  llvm::DenseMap<const void *, bool> analyzedFunctions;
  llvm::DenseMap<const clang::Stmt *, unsigned> loopTags;
  llvm::DenseMap<const clang::VarDecl *, unsigned> storageTags;
  llvm::DenseMap<const clang::VarDecl *, llvm::SmallVector<unsigned, 1>>
      assignTags;
  llvm::DenseMap<const clang::VarDecl *, GlobalTarget> globals;
  llvm::DenseMap<const clang::VarDecl *, mlir::Operation *> globalOps;
  /// Warnings for bind_op pragmas whose operation was never seen.
  bool checkedTU = false;
  /// True while running checkAllFunctions (no codegen, no materialization).
  bool checkOnlyMode = false;

  // Persistent across translation units (consumed after the pipeline).
  std::vector<TagRecord> tags;
  /// The tag that marks the stores of an inlined ap_fixed compound
  /// assignment (hls.ap_compound_assign); created on first use.
  std::optional<unsigned> compoundAssignTag;
  std::vector<ReportEntry> report;

  /// Records an outcome for the pragma's report entry.
  void addOutcome(int reportIndex, std::string outcome) {
    if (reportIndex >= 0)
      report[reportIndex].outcomes.push_back(std::move(outcome));
  }
  bool hasOutcome(int reportIndex) const {
    return reportIndex >= 0 && !report[reportIndex].outcomes.empty();
  }

  void resetTranslationUnit();

  TagRecord &newTag(TagClass cls);
  mlir::Location tag(mlir::Location base, unsigned tagId) const;

  // Diagnostics (valid during the translation unit only).
  clang::DiagnosticBuilder error(clang::SourceLocation loc,
                                 llvm::StringRef fmt);
  clang::DiagnosticBuilder warning(clang::SourceLocation loc,
                                   llvm::StringRef fmt);
  clang::DiagnosticBuilder note(clang::SourceLocation loc,
                                llvm::StringRef fmt);
  clang::DiagnosticBuilder remark(clang::SourceLocation loc,
                                  llvm::StringRef fmt);
  std::string locString(clang::SourceLocation loc) const;
};

/// Name of the location-tag metadata key.
constexpr llvm::StringLiteral kTagKey = "hls.tag";

/// Whether --hls-pragma-report is on.
bool reportEnabled();

/// Registers the catch-all pragma handler (HLSPragmaParse.cc).
void registerPragmaHandler(clang::Preprocessor &PP);

/// `#pragma HLS <kind>` as written in diagnostics.
std::string pragmaDisplayName(const PragmaRecord &P);

} // namespace hls
} // namespace cgeist

#endif // CGEIST_LIB_HLSPRAGMAINTERNAL_H
