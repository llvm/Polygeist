//===- HLSAutoInline.h - Vitis automatic inlining decision ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Vitis HLS 2025.1 inlines a callee without an inline pragma automatically
// ([HLS 214-178]) when it is small, and keeps it as its own module otherwise.
// The rule, pinned by kernels in the hls-est tree (VITIS_NOTES N18, MEMOS
// decision #34):
//
//  * The call graph is visited bottom-up. For a caller C and a callee F
//    without an inline pragma, all calls of F in C are inlined iff
//        size(F) x calls(F in C) <= 100,
//    and none of them otherwise. Calls are counted after the user's unroll
//    pragmas and the complete unroll a pipeline pragma implies for every
//    loop nested in the pipelined loop or function ([HLS 214-291]).
//  * size(F) is F's instruction count after Vitis's simplification, without
//    the integer casts (sext / zext / trunc), plus size(G) - 2 for every call
//    of a G inlined into F before.
//  * A callee that returns a pointer or a reference is always inlined; an
//    `inline off` callee, a process of a dataflow region (a call at the top
//    level of a `dataflow` function, also inside a loop of it that a
//    complete unroll pragma with constant bounds removes; a call inside
//    another loop of it follows the size rule) and a callee that holds a
//    dataflow region are always kept; an `inline` (on) callee and an
//    `always_inline` one (clang inlines it first; the Vitis headers mark
//    every ap_int / ap_fixed member so) are always inlined, and the calls in
//    an always_inline body count in the caller it is inlined into.
//  * A dataflow pragma counts only directly in a function body or a `for`
//    body; elsewhere Vitis ignores it ([HLS 207-5576]; HLSPragmaSema.cc).
//
// cgeist measures size(F) the way N18's proxy does (146 of 150 decisions of
// the N18 kernels): clang's own LLVM IR for the translation unit, at -O0
// without optnone, the always_inline callees inlined (as clang -O0 does),
// with the loops above unrolled first, then `sroa, early-cse, instcombine,
// simplifycfg, dce`. A consumer
// next to cgeist's MLIR codegen builds that IR; a pass right before cgeist's
// inliner applies the rule and marks the calls Vitis keeps
// (`hls.inline_auto_off`, on the func.call and on the callee); cgeist's
// inliner interface does not inline them (HLSPragma.cc), and hls-est treats
// them like `inline off` calls.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_LIB_HLSAUTOINLINE_H
#define CGEIST_LIB_HLSAUTOINLINE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace clang {
class ASTConsumer;
class CompilerInstance;
class SourceLocation;
class SourceManager;
} // namespace clang

namespace mlir {
class Pass;
} // namespace mlir

namespace cgeist {
namespace hls {

/// `hls.inline_auto_off`: on a func.call, Vitis keeps this call (dictionary
/// {size, calls}); on a func.func, at least one of its calls is kept that way
/// (dictionary {size}).
constexpr llvm::StringLiteral kInlineAutoOffAttr = "hls.inline_auto_off";

/// Whether the automatic-inlining decision runs (--hls-annotate and
/// --hls-auto-inline, the default).
bool autoInlineEnabled();

/// A consumer that generates clang's LLVM IR for the translation unit next to
/// cgeist's own codegen and measures every function when the translation unit
/// ends. Add it after cgeist's consumer (it reads the unroll pragmas that
/// consumer's pragma sema records).
std::unique_ptr<clang::ASTConsumer>
createInlineSizeConsumer(clang::CompilerInstance &CI, llvm::StringRef inFile);

/// Records a loop that carries `#pragma HLS unroll` (factor 0 = complete,
/// -1 = `unroll off`).
void recordUnrollLoop(const clang::SourceManager &SM, clang::SourceLocation loop,
                      int64_t factor);

/// Records a loop that carries `#pragma HLS pipeline` (not `off`): Vitis
/// unrolls every loop nested in it completely before it sizes functions for
/// automatic inlining ([HLS 214-291], VITIS_NOTES N18).
void recordPipelineLoop(const clang::SourceManager &SM,
                        clang::SourceLocation loop);

/// Records a function (its symbol) whose body carries `#pragma HLS pipeline`:
/// every loop in it is unrolled completely, as under a pipelined loop.
void recordPipelineFunction(llvm::StringRef symbol);

/// The pass that applies the rule and marks the kept calls; add it right
/// before cgeist's inliner.
std::unique_ptr<mlir::Pass> createMarkAutoInlinePass();

} // namespace hls
} // namespace cgeist

#endif // CGEIST_LIB_HLSAUTOINLINE_H
