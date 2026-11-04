//===- HLSPragma.h - Vitis HLS pragma extraction ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public interface of the `#pragma HLS` extraction used by cgeist codegen and
// the driver. The design is documented in docs/hls-pragmas.md; in short:
//
//  1. Parsing (preprocessor): a catch-all pragma handler recognises the `HLS`
//     namespace case-insensitively and turns each pragma into a typed record.
//  2. Association (AST): when a function body is about to be emitted, every
//     pragma inside it is bound to its target (loop statement, function,
//     parameter, local/global variable) by source location, validated, and
//     turned into `hls.*` attributes.
//  3. Emission: attributes of functions and parameters are set directly.
//     Loops, local variables and `bind_op` operations do not have a stable op
//     at codegen time (loops are still a CFG), so the op is given a *tagged
//     location* (a FusedLoc whose metadata names the target). Locations are
//     carried along by every rewrite, clone and inline.
//  4. Materialization (after the pass pipeline): tags are turned into the
//     final attributes on whichever op carries them, and removed from the
//     locations.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_LIB_HLSPRAGMA_H
#define CGEIST_LIB_HLSPRAGMA_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include <memory>

namespace clang {
class ASTContext;
class CXXRecordDecl;
class FunctionDecl;
class Preprocessor;
class Stmt;
class VarDecl;
} // namespace clang

namespace mlir {
class DialectRegistry;
class Pass;
class MLIRContext;
class ModuleOp;
class Operation;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace cgeist {
namespace hls {

/// Whether `--hls-annotate` is on.
bool isEnabled();

/// Whether `--hls-pragma-check-only` is on: pragmas of every function body
/// are parsed, associated and validated, and no code is generated.
bool checkOnly();

//===-- Preprocessor stage -----------------------------------------------===//

/// Registers the HLS pragma handler on `PP` and starts a new translation
/// unit. Always called; pragmas are only interpreted when isEnabled().
void beginTranslationUnit(clang::Preprocessor &PP, mlir::MLIRContext *ctx);

//===-- AST / codegen stage ----------------------------------------------===//

/// Diagnoses pragmas that are not inside any function body. Called once
/// after the whole translation unit has been parsed.
void checkTranslationUnit(clang::ASTContext &astCtx);

/// Check-only mode: associates and validates the pragmas of every function
/// definition in the translation unit (template instantiations included)
/// without generating code; outcomes go to the report.
void checkAllFunctions(clang::ASTContext &astCtx);

/// Associates and validates all pragmas inside the body of `FD` and sets the
/// function-level and parameter-level attributes on `func`. `argOffset` is
/// the index of the first source parameter among the function's arguments
/// (1 for instance methods, which take `this` first).
void enterFunction(const clang::FunctionDecl *FD, mlir::func::FuncOp func,
                   unsigned argOffset);

/// Returns `base`, tagged if `loop` is the target of any pragma. Used for
/// the op that evaluates the loop's exit condition (or the affine.for).
mlir::Location loopLocation(const clang::Stmt *loop, mlir::Location base);

/// Called for the op that provides storage of a local variable (alloca).
void onLocalStorage(const clang::VarDecl *VD, mlir::Operation *storage);

/// Called when a global (namespace-scope or static local) is created.
void onGlobal(const clang::VarDecl *VD, mlir::Operation *global);

/// Called with the value assigned to `VD` (by `=`, `op=` or an initializer);
/// used by `bind_op`.
void onAssignedValue(const clang::VarDecl *VD, mlir::Value value);

/// Finishes the translation unit: applies attributes to globals and reports
/// pragmas whose target never materialized during codegen.
void endTranslationUnit();

//===-- MLIR stage -------------------------------------------------------===//

/// Turns location tags into `hls.*` attributes and strips the tags. When
/// `reportLost` is set, warns about tagged targets that no longer exist
/// (e.g. a fully unrolled loop or a variable promoted to registers).
void materialize(mlir::ModuleOp module, bool reportLost);

/// An AGGREGATE pragma on a parameter (or on `this` of a member function)
/// reaches the variable a call passes to it, as in Vitis (VITIS_NOTES N37
/// S2 (b), (c) in the hls-est tree): every call to a function with an
/// `hls.aggregate` argument marks the alloca, memref.global or caller
/// argument the actual is a whole-element view of, transitively, with the
/// callee's entry plus `from = <callee>`. A variable that already carries
/// `hls.aggregate` keeps its own. Off with --hls-propagate-aggregate=false.
void propagateAggregate(mlir::ModuleOp module);

/// propagateAggregate() as a pass; add it before cgeist's inliner, which
/// removes the calls that carry the information.
std::unique_ptr<mlir::Pass> createPropagateAggregatePass();

//===-- Inlining ---------------------------------------------------------===//

/// Registers cgeist's inliner interface for the func dialect, in place of
/// mlir::func::registerInlinerExtension. It is MLIR's interface with one
/// rule added: a call to a function marked `#pragma HLS inline off`
/// (attribute `hls.inline_off`, set with --hls-annotate) is not inlined, so
/// the call stays a func.call as in Vitis, which keeps such a function as
/// its own module. --hls-ignore-inline-off restores the former behaviour
/// (every call is inlined). A dialect keeps the first interface registered
/// for it, so this must run before mlir::registerAllExtensions.
void registerInlinerExtension(mlir::DialectRegistry &registry);

/// Whether the func dialect of `ctx` uses the interface registered by
/// registerInlinerExtension() (loads the func dialect).
/// --xlx-headers (HLSPragma.cc).
bool xlxHeaders();

/// Whether `rd` is an ap_fixed / ap_ufixed / ap_fixed_base class of the
/// Vitis headers, or derives from one.
bool isApFixedClass(const clang::CXXRecordDecl *rd);

bool inlinerExtensionActive(mlir::MLIRContext *ctx);

/// --hls-keep-misfit-inline-off: an `inline off` call whose array argument
/// does not fit the parameter's first dimension is kept (the argument
/// reinterpreted to the parameter type).
bool keepMisfitInlineOff();

} // namespace hls
} // namespace cgeist

#endif // CGEIST_LIB_HLSPRAGMA_H
