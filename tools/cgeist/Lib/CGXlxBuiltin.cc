// (C) Copyright 2016-2022 Xilinx, Inc.
// (C) Copyright 2023-2025 Advanced Micro Devices, Inc.
// All Rights Reserved.
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//===----------------------------------------------------------------------===//
//
// cgeist lowering of the Xilinx HLS bit builtins that the Vitis HLS headers
// (common/technology/autopilot/etc/ap_common.h, autopilot_ssdm_bits.h,
// ap_int_base.h) call under __SYNTHESIS__.
//
// Origin: ported from hls-llvm-project (github.com/Xilinx/hls-llvm-project)
// commit ee930469, clang/lib/CodeGen/CGFPGABuiltin.cpp:1000-1677
// (EmitBuiltinBitConcat, getRadix, EmitBuiltinBitFromString,
// EmitBuiltinBitSelect, EmitBuiltinBitSet, EmitBuiltinBitPartSelect,
// EmitBuiltinBitPartSet, EmitBuiltinBit{And,NAnd,Or,NOr,Xor,NXor}Reduce) and
// :1870-1905 (EmitBuiltinFPGACount{Leading,Trailing}Zero), under the Apache
// License 2.0 shown above.
//
// Changes from the original (Polygeist / cgeist, track W11, 2026-10):
//  - Emits MLIR (arith / math) instead of LLVM IR; the bit reversal of a
//    reversed range is expanded into shift / mask stages.
//  - The operands' widths are recovered from the `(void*)&x` argument's
//    pointee type in the clang AST instead of from the LLVM bitcast operand.
//  - llvm.fpga.legacy.part.select / part.set are not emitted; their semantics
//    are expanded into shift / mask / select ops. The expansion follows the
//    constant folds of those intrinsics (hls-llvm-project ee930469,
//    llvm/lib/Analysis/ConstantFolding.cpp:1368-1437, a University of
//    Illinois/NCSA licensed file with Xilinx additions; re-implemented from
//    its documented behaviour, not copied): bits [min(Lo,Hi), max(Lo,Hi)] are
//    selected / replaced, and the field is bit-reversed when Lo > Hi.
//    Lo or Hi outside the operand width is not folded by Vitis (its
//    behaviour is unknown); here it gives the shift-and-mask result, which is
//    unconfirmed.
//  - The builtins are found by name; they are declared by cgeist's prelude
//    (include/xlx/xlx_prelude.h) instead of Builtins.def.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang-mlir.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

bool isXlxBuiltinName(llvm::StringRef name) {
  return name.starts_with("__builtin_bit_") || name == "__fpga_ctlz" ||
         name == "__fpga_cttz" || name == "__fpga_shift_register_peek" ||
         name == "__fpga_shift_register_shift";
}

/// Port of getRadix (CGFPGABuiltin.cpp:1065): the radix of a decimal-radix
/// string with a 0 / 0x / 0b prefix; strips the prefix from `s`. 0: cannot
/// be processed. Unlike the original, `s` has no trailing NUL here (the
/// original's `length() - 1` counted it), and an empty digit string after a
/// prefix returns 0 instead of asserting.
uint32_t getRadix(std::string &s) {
  std::string prefix = "";
  if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
    prefix += s[0];
    s = s.substr(1);
  }
  size_t len = s.length();
  if (len == 0)
    return 0;
  if (len == 1) {
    if (s[0] <= '9' && s[0] >= '0') {
      s = prefix + s;
      return 10;
    }
    return 0; // only one char and it is not decimal
  }
  if (s[0] != '0') {
    s = prefix + s;
    return 10;
  }
  if (s[1] == 'x' || s[1] == 'X') {
    if (len == 2)
      return 0; // nothing follows the radix
    s = prefix + s.substr(2);
    return 16;
  }
  if (s[1] == 'b' || s[1] == 'B') {
    if (len == 2)
      return 0;
    s = prefix + s.substr(2);
    return 2;
  }
  s = prefix + s.substr(1);
  return 8;
}

} // namespace

std::pair<ValueCategory, bool>
MLIRScanner::EmitXlxBuiltin(clang::CallExpr *expr) {
  const clang::FunctionDecl *FD = expr->getDirectCallee();
  if (!FD || !FD->getIdentifier() || isa<clang::CXXMethodDecl>(FD) ||
      !isXlxBuiltinName(FD->getName()))
    return std::make_pair(ValueCategory(), false);
  llvm::StringRef name = FD->getName();
  mlir::Location loc = getMLIRLocation(expr->getExprLoc());
  clang::DiagnosticsEngine &Diags = Glob.CGM.getDiags();
  auto done = [](ValueCategory v) { return std::make_pair(v, true); };

  auto error = [&](const clang::Expr *at, llvm::StringRef msg) {
    Diags.Report(at->getExprLoc(),
                 Diags.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0"))
        << msg;
  };
  // Result of an int-returning builtin, or a placeholder after an error.
  auto intResult = [&](mlir::Value bit) -> ValueCategory {
    mlir::Type dest = getMLIRType(expr->getType());
    auto destTy = dest.dyn_cast<mlir::IntegerType>();
    if (!destTy)
      return ValueCategory();
    if (!bit)
      return ValueCategory(
          builder.create<arith::ConstantIntOp>(loc, 0, destTy), false);
    return ValueCategory(builder.create<arith::ExtUIOp>(loc, destTy, bit),
                         false);
  };
  auto fail = [&](const clang::Expr *at, llvm::StringRef msg) {
    error(at, msg);
    return done(intResult(nullptr));
  };

  auto constant = [&](mlir::IntegerType ty, const llvm::APInt &v) {
    return builder
        .create<arith::ConstantOp>(loc, ty, builder.getIntegerAttr(ty, v))
        .getResult();
  };
  auto constantU = [&](mlir::IntegerType ty, uint64_t v) {
    return constant(ty, llvm::APInt(64, v).zextOrTrunc(ty.getWidth()));
  };
  // Zero-extend or truncate (CreateIntCast(..., /*isSigned=*/false)).
  auto castTo = [&](mlir::Value v, mlir::IntegerType ty) -> mlir::Value {
    unsigned from = v.getType().cast<mlir::IntegerType>().getWidth();
    if (from == ty.getWidth())
      return v;
    if (from < ty.getWidth())
      return builder.create<arith::ExtUIOp>(loc, ty, v);
    return builder.create<arith::TruncIOp>(loc, ty, v);
  };

  // The lvalue behind a `(void*)(&x)` argument (casts and parentheses
  // stripped); the operand width is the width of its storage type, as with
  // the original's pointer element type.
  auto pointee = [&](clang::Expr *arg, ValueCategory &lv,
                     mlir::IntegerType &ty) -> bool {
    auto *uo = dyn_cast<clang::UnaryOperator>(arg->IgnoreParenCasts());
    if (!uo || uo->getOpcode() != clang::UO_AddrOf)
      return false;
    ty = getMLIRType(uo->getSubExpr()->getType())
             .dyn_cast_or_null<mlir::IntegerType>();
    if (!ty)
      return false;
    lv = Visit(uo->getSubExpr());
    return lv.val && lv.isReference;
  };
  auto intArg = [&](clang::Expr *arg) -> mlir::Value {
    mlir::Value v = Visit(arg).getValue(loc, builder);
    if (!v || !v.getType().isa<mlir::IntegerType>())
      return nullptr;
    return castTo(v, builder.getI32Type());
  };

  // A field [min(lo,hi), max(lo,hi)] of an iN value: the pieces shared by
  // part_select and part_set.
  struct Field {
    mlir::Value reversed; ///< i1: lo > hi
    mlir::Value low;      ///< iN: min(lo, hi)
    mlir::Value mask;     ///< iN: n low ones, n = |hi - lo| + 1
    mlir::Value unusedHi; ///< iN: N - n
  };
  auto field = [&](mlir::IntegerType ty, mlir::Value lo32,
                   mlir::Value hi32) -> Field {
    Field F;
    F.reversed = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               lo32, hi32);
    mlir::Value l =
        builder.create<arith::SelectOp>(loc, F.reversed, hi32, lo32);
    mlir::Value h =
        builder.create<arith::SelectOp>(loc, F.reversed, lo32, hi32);
    mlir::Value n32 = builder.create<arith::AddIOp>(
        loc, builder.create<arith::SubIOp>(loc, h, l),
        constantU(builder.getI32Type(), 1));
    F.low = castTo(l, ty);
    F.unusedHi = builder.create<arith::SubIOp>(
        loc, constantU(ty, ty.getWidth()), castTo(n32, ty));
    F.mask = builder.create<arith::ShRUIOp>(
        loc, constant(ty, llvm::APInt::getAllOnes(ty.getWidth())),
        F.unusedHi);
    return F;
  };
  // Bit-reverse the n low bits of v (bits above n are zero). Built from
  // arith only (log2 swap stages at the next power-of-two width P), so that
  // hls-est needs no new op and constant operands fold.
  auto reverseLow = [&](mlir::Value v, const Field &F) -> mlir::Value {
    auto ty = v.getType().cast<mlir::IntegerType>();
    unsigned N = ty.getWidth();
    unsigned P = llvm::PowerOf2Ceil(N);
    auto pty = builder.getIntegerType(P);
    mlir::Value r = castTo(v, pty);
    for (unsigned s = 1; s < P; s *= 2) {
      // s ones, s zeros, repeated: 0x55.., 0x33.., 0x0F.., ...
      llvm::APInt m(P, 0);
      for (unsigned b = 0; b < P; ++b)
        if ((b / s) % 2 == 0)
          m.setBit(b);
      mlir::Value mask = constant(pty, m);
      mlir::Value sh = constantU(pty, s);
      mlir::Value hi = builder.create<arith::AndIOp>(
          loc, builder.create<arith::ShRUIOp>(loc, r, sh), mask);
      mlir::Value lo = builder.create<arith::ShLIOp>(
          loc, builder.create<arith::AndIOp>(loc, r, mask), sh);
      r = builder.create<arith::OrIOp>(loc, hi, lo);
    }
    // r holds the P-bit reversal; the n-bit field reversed is r >> (P - n).
    mlir::Value amount = castTo(F.unusedHi, pty);
    if (P != N)
      amount = builder.create<arith::AddIOp>(loc, amount,
                                             constantU(pty, P - N));
    r = builder.create<arith::ShRUIOp>(loc, r, amount);
    return castTo(r, ty);
  };
  auto partSelect = [&](mlir::Value val, mlir::Value lo32,
                        mlir::Value hi32) -> mlir::Value {
    auto ty = val.getType().cast<mlir::IntegerType>();
    Field F = field(ty, lo32, hi32);
    mlir::Value fwd = builder.create<arith::AndIOp>(
        loc, builder.create<arith::ShRUIOp>(loc, val, F.low), F.mask);
    return builder.create<arith::SelectOp>(loc, F.reversed,
                                           reverseLow(fwd, F), fwd);
  };
  auto partSet = [&](mlir::Value val, mlir::Value rep, mlir::Value lo32,
                     mlir::Value hi32) -> mlir::Value {
    auto ty = val.getType().cast<mlir::IntegerType>();
    Field F = field(ty, lo32, hi32);
    // Rep is cut or zero-extended to the field width (zextOrTrunc).
    mlir::Value r =
        builder.create<arith::AndIOp>(loc, castTo(rep, ty), F.mask);
    r = builder.create<arith::SelectOp>(loc, F.reversed, reverseLow(r, F), r);
    mlir::Value hole = builder.create<arith::XOrIOp>(
        loc, builder.create<arith::ShLIOp>(loc, F.mask, F.low),
        constant(ty, llvm::APInt::getAllOnes(ty.getWidth())));
    return builder.create<arith::OrIOp>(
        loc, builder.create<arith::AndIOp>(loc, val, hole),
        builder.create<arith::ShLIOp>(loc, r, F.low));
  };

  auto checkArgs = [&](unsigned n) { return expr->getNumArgs() == n; };
  std::string argError = "invalid arguments to '" + name.str() + "'";

  // __builtin_bit_concat(R*, H*, L*): R = {H, L}.
  if (name == "__builtin_bit_concat") {
    ValueCategory R, H, L;
    mlir::IntegerType RTy, HTy, LTy;
    if (!checkArgs(3) || !pointee(expr->getArg(0), R, RTy) ||
        !pointee(expr->getArg(1), H, HTy) ||
        !pointee(expr->getArg(2), L, LTy))
      return fail(expr, "All three arguments to __builtin_bit_concat must be "
                        "int* typed");
    if (RTy.getWidth() != HTy.getWidth() + LTy.getWidth())
      return fail(expr->getArg(0),
                  "Invalid bit width for __builtin_bit_concat result");
    mlir::Value hi = castTo(H.getValue(loc, builder), RTy);
    mlir::Value lo = castTo(L.getValue(loc, builder), RTy);
    hi = builder.create<arith::ShLIOp>(loc, hi,
                                       constantU(RTy, LTy.getWidth()));
    R.store(loc, builder, builder.create<arith::OrIOp>(loc, hi, lo));
    return done(ValueCategory());
  }

  // __builtin_bit_from_string(R*, "literal", radix): folded at compile time.
  if (name == "__builtin_bit_from_string") {
    ValueCategory R;
    mlir::IntegerType RTy;
    if (!checkArgs(3) || !pointee(expr->getArg(0), R, RTy))
      return fail(expr, "First argument to __builtin_bit_from_string must be "
                        "int* typed");
    auto *lit =
        dyn_cast<clang::StringLiteral>(expr->getArg(1)->IgnoreParenCasts());
    if (!lit || !lit->isOrdinary())
      return fail(expr->getArg(1), "Second argument to "
                                   "__builtin_bit_from_string must be a "
                                   "constant C string");
    clang::Expr::EvalResult radixEval;
    if (!expr->getArg(2)->EvaluateAsInt(radixEval, Glob.CGM.getContext()) ||
        radixEval.Val.getInt().isZero())
      return fail(expr->getArg(2), "Third argument to "
                                   "__builtin_bit_from_string must be a "
                                   "constant int");
    uint64_t radix = radixEval.Val.getInt().getZExtValue();
    if (radix != 2 && radix != 8 && radix != 10 && radix != 16)
      return fail(expr->getArg(2), "Third argument to "
                                   "__builtin_bit_from_string must be 2, 8, "
                                   "10 or 16");
    std::string str = lit->getString().str();
    if (radix == 10)
      radix = getRadix(str);
    if (radix == 0 || str.empty())
      return fail(expr->getArg(1), "Can not process radix for second "
                                   "arguments of __builtin_bit_from_string");
    unsigned bits = RTy.getWidth();
    unsigned needed = llvm::APInt::getBitsNeeded(str, radix);
    llvm::APInt v(needed, str, radix);
    if (bits < needed)
      v = v.trunc(bits);
    else if (bits > needed)
      v = str[0] == '-' ? v.sext(bits) : v.zext(bits);
    R.store(loc, builder, constant(RTy, v));
    return done(ValueCategory());
  }

  // __builtin_bit_select(V*, bit): bit `bit` of V, zero-extended to int.
  if (name == "__builtin_bit_select") {
    ValueCategory V;
    mlir::IntegerType VTy;
    if (!checkArgs(2) || !pointee(expr->getArg(0), V, VTy))
      return fail(expr, "First argument to __builtin_bit_select must be int* "
                        "typed");
    mlir::Value bit = intArg(expr->getArg(1));
    if (!bit)
      return fail(expr->getArg(1), "Second argument to __builtin_bit_select "
                                   "must be an integer type");
    mlir::Value v = V.getValue(loc, builder);
    mlir::Value sh = builder.create<arith::ShRUIOp>(loc, v, castTo(bit, VTy));
    return done(intResult(
        builder.create<arith::TruncIOp>(loc, builder.getI1Type(), sh)));
  }

  // __builtin_bit_set(R*, V*, Rep*, bit) and
  // __builtin_bit_part_set(R*, V*, Rep*, lo, hi): R = V with the field
  // replaced by Rep.
  if (name == "__builtin_bit_set" || name == "__builtin_bit_part_set") {
    bool single = name == "__builtin_bit_set";
    ValueCategory R, V, Rep;
    mlir::IntegerType RTy, VTy, RepTy;
    if (!checkArgs(single ? 4 : 5) || !pointee(expr->getArg(0), R, RTy) ||
        !pointee(expr->getArg(1), V, VTy) ||
        !pointee(expr->getArg(2), Rep, RepTy))
      return fail(expr, argError + ": the first three must be int* typed");
    mlir::Value lo = intArg(expr->getArg(3));
    mlir::Value hi = single ? lo : intArg(expr->getArg(4));
    if (!lo || !hi)
      return fail(expr, argError + ": the bit indices must be int typed");
    mlir::Value res = partSet(V.getValue(loc, builder),
                              Rep.getValue(loc, builder), lo, hi);
    R.store(loc, builder, castTo(res, RTy));
    return done(ValueCategory());
  }

  // __builtin_bit_part_select(R*, V*, lo, hi): R = V[hi:lo] (reversed when
  // lo > hi), zero-extended.
  if (name == "__builtin_bit_part_select") {
    ValueCategory R, V;
    mlir::IntegerType RTy, VTy;
    if (!checkArgs(4) || !pointee(expr->getArg(0), R, RTy) ||
        !pointee(expr->getArg(1), V, VTy))
      return fail(expr, argError + ": the first two must be int* typed");
    mlir::Value lo = intArg(expr->getArg(2));
    mlir::Value hi = intArg(expr->getArg(3));
    if (!lo || !hi)
      return fail(expr, argError + ": the bit indices must be int typed");
    mlir::Value res = partSelect(V.getValue(loc, builder), lo, hi);
    R.store(loc, builder, castTo(res, RTy));
    return done(ValueCategory());
  }

  // Reductions: __builtin_bit_{and,nand,or,nor,xor,nxor}_reduce(V*).
  if (name.ends_with("_reduce")) {
    llvm::StringRef op = name.drop_front(strlen("__builtin_bit_"))
                             .drop_back(strlen("_reduce"));
    if (op != "and" && op != "nand" && op != "or" && op != "nor" &&
        op != "xor" && op != "nxor")
      return fail(expr, "unsupported Xilinx HLS builtin '" + name.str() + "'");
    ValueCategory V;
    mlir::IntegerType VTy;
    if (!checkArgs(1) || !pointee(expr->getArg(0), V, VTy))
      return fail(expr, argError + ": the argument must be int* typed");
    mlir::Value v = V.getValue(loc, builder);
    mlir::Value bit;
    if (op == "and" || op == "nand") {
      mlir::Value allOnes =
          constant(VTy, llvm::APInt::getAllOnes(VTy.getWidth()));
      bit = builder.create<arith::CmpIOp>(loc,
                                          op == "and"
                                              ? arith::CmpIPredicate::eq
                                              : arith::CmpIPredicate::ne,
                                          v, allOnes);
    } else if (op == "or" || op == "nor") {
      bit = builder.create<arith::CmpIOp>(loc,
                                          op == "or"
                                              ? arith::CmpIPredicate::ne
                                              : arith::CmpIPredicate::eq,
                                          v, constantU(VTy, 0));
    } else {
      mlir::Value pop = builder.create<math::CtPopOp>(loc, v);
      bit = builder.create<arith::TruncIOp>(loc, builder.getI1Type(), pop);
      if (op == "nxor")
        bit = builder.create<arith::XOrIOp>(
            loc, bit, constantU(builder.getI1Type(), 1));
    }
    return done(intResult(bit));
  }

  // __fpga_ctlz(x) / __fpga_cttz(x): count at the operand's own width, as an
  // unsigned int (CGFPGABuiltin.cpp:1870-1905: an implicit conversion to int
  // on the argument is looked through; zero gives the width).
  if (name == "__fpga_ctlz" || name == "__fpga_cttz") {
    if (!checkArgs(1))
      return fail(expr, argError);
    clang::Expr *src = expr->getArg(0);
    if (auto *ice = dyn_cast<clang::ImplicitCastExpr>(src))
      if (const auto *BT = ice->getType()->getAs<clang::BuiltinType>();
          BT && BT->getKind() == clang::BuiltinType::Int)
        src = ice->getSubExpr();
    mlir::Value v = Visit(src).getValue(loc, builder);
    if (!v || !v.getType().isa<mlir::IntegerType>())
      return fail(expr, argError + ": the argument must be an integer");
    mlir::Value count =
        name == "__fpga_ctlz"
            ? builder.create<math::CountLeadingZerosOp>(loc, v).getResult()
            : builder.create<math::CountTrailingZerosOp>(loc, v).getResult();
    count = castTo(count, builder.getI32Type());
    if (auto destTy =
            getMLIRType(expr->getType()).dyn_cast<mlir::IntegerType>())
      count = castTo(count, destTy);
    return done(ValueCategory(count, false));
  }

  // ap_shift_reg (ap_shift_reg.h): `peek(Array, Addr)` reads Array[Addr];
  // `shift(DataIn, Array, Enable)` moves every element one place up
  // (Array[i] = Array[i - 1]) and stores DataIn at 0 when Enable. The depth
  // is the extent of the array the pointer argument is cast from (the
  // header passes `(T *)Array` of `T Array[DEPTH]`). CIRCT OPEN_ISSUES O219
  // (Vitis_Libraries dsp 1-D FFT).
  if (name == "__fpga_shift_register_peek" ||
      name == "__fpga_shift_register_shift") {
    bool peek = name == "__fpga_shift_register_peek";
    if (expr->getNumArgs() != (peek ? 2u : 3u))
      return fail(expr, "wrong number of arguments to '" + name.str() + "'");
    clang::Expr *arrayArg = expr->getArg(peek ? 0 : 1);
    ValueCategory ptr = Visit(arrayArg);
    if (!ptr.val)
      return fail(arrayArg, "cannot lower the shift register's array");
    if (peek) {
      mlir::Value addr = Visit(expr->getArg(1)).getValue(loc, builder);
      if (!addr || !addr.getType().isa<mlir::IntegerType>())
        return fail(expr->getArg(1), "the address is not an integer");
      ValueCategory el =
          CommonArrayLookup(loc, ptr, castToIndex(loc, addr, true), false);
      return done(ValueCategory(el.getValue(loc, builder), false));
    }
    const clang::ConstantArrayType *cat =
        Glob.CGM.getContext().getAsConstantArrayType(
            arrayArg->IgnoreParenCasts()->getType());
    if (!cat)
      return fail(arrayArg, "the shift register is not a constant-size array");
    int64_t depth = cat->getSize().getSExtValue();
    mlir::Value in = Visit(expr->getArg(0)).getValue(loc, builder);
    mlir::Value en = Visit(expr->getArg(2)).getValue(loc, builder);
    if (!in || !en || !en.getType().isa<mlir::IntegerType>())
      return fail(expr, "cannot lower the shift register's operands");
    auto enTy = en.getType().cast<mlir::IntegerType>();
    mlir::Value cond = enTy.getWidth() == 1
                           ? en
                           : builder.create<arith::CmpIOp>(
                                 loc, arith::CmpIPredicate::ne, en,
                                 builder.create<arith::ConstantIntOp>(
                                     loc, 0, enTy));
    auto ifOp = builder.create<scf::IfOp>(loc, cond, /*withElse=*/false);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (depth > 1) {
      // for k in [0, depth - 1): i = depth - 1 - k; a[i] = a[i - 1]
      auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto last = builder.create<arith::ConstantIndexOp>(loc, depth - 1);
      auto forOp = builder.create<scf::ForOp>(loc, c0, last, c1);
      builder.setInsertionPointToStart(forOp.getBody());
      mlir::Value i = builder.create<arith::SubIOp>(
          loc, last, forOp.getInductionVar());
      mlir::Value prev = builder.create<arith::SubIOp>(loc, i, c1);
      mlir::Value v =
          CommonArrayLookup(loc, ptr, prev, false).getValue(loc, builder);
      CommonArrayLookup(loc, ptr, i, false).store(loc, builder, v);
      builder.setInsertionPointAfter(forOp);
    }
    CommonArrayLookup(loc, ptr, builder.create<arith::ConstantIndexOp>(loc, 0),
                      false)
        .store(loc, builder, in);
    return done(ValueCategory());
  }

  return fail(expr, "unsupported Xilinx HLS builtin '" + name.str() + "'");
}
