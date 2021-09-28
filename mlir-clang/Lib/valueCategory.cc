//===- valueCategory.cc ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "valueCategory.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "polygeist/Ops.h"

using namespace mlir;

mlir::Value ValueWithOffsets::getValue(mlir::OpBuilder &builder) const {
  assert(val && "must be not-null");
  if (!isReference)
    return val;
  auto loc = builder.getUnknownLoc();
  if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
    return builder.create<mlir::LLVM::LoadOp>(loc, val);
  }
  if (auto mt = val.getType().dyn_cast<mlir::MemRefType>()) {
    assert(mt.getShape().size() == 1 && "must have shape 1");
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    return builder.create<memref::LoadOp>(loc, val,
                                          std::vector<mlir::Value>({c0}));
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

void ValueWithOffsets::store(mlir::OpBuilder &builder,
                             mlir::Value toStore) const {
  assert(isReference && "must be a reference");
  assert(val && "expect not-null");
  auto loc = builder.getUnknownLoc();
  if (auto pt = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (toStore.getType() != pt.getElementType()) {
      if (auto mt = toStore.getType().dyn_cast<MemRefType>()) {
        if (auto spt =
                pt.getElementType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
          assert(mt.getElementType() == spt.getElementType() &&
                 "expect same type");
          toStore =
              builder.create<polygeist::Memref2PointerOp>(loc, spt, toStore);
        }
      }
    } else { // toStore.getType() == pt.getElementType()
      assert(toStore.getType() == pt.getElementType() && "expect same type");
      builder.create<mlir::LLVM::StoreOp>(loc, toStore, val);
    }
    return;
  }
  if (auto mt = val.getType().dyn_cast<MemRefType>()) {
    assert(mt.getShape().size() == 1 && "must have size 1");
    assert(toStore.getType() ==
               val.getType().cast<MemRefType>().getElementType() &&
           "expect same type");
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    builder.create<mlir::memref::StoreOp>(loc, toStore, val,
                                          std::vector<mlir::Value>({c0}));
    return;
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

ValueWithOffsets ValueWithOffsets::dereference(mlir::OpBuilder &builder) const {
  assert(val && "val must be not-null");

  auto loc = builder.getUnknownLoc();
  if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
    if (!isReference)
      return ValueWithOffsets(val, /*isReference*/ true);
    else
      return ValueWithOffsets(builder.create<mlir::LLVM::LoadOp>(loc, val),
                              /*isReference*/ true);
  }

  if (auto mt = val.getType().cast<mlir::MemRefType>()) {
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    auto shape = std::vector<int64_t>(mt.getShape());

    if (isReference) {
      if (shape.size() > 1) {
        shape.erase(shape.begin());
        auto mt0 =
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  mt.getAffineMaps(), mt.getMemorySpace());
        return ValueWithOffsets(
            builder.create<polygeist::SubIndexOp>(loc, mt0, val, c0),
            /*isReference*/ true);
      } else {
        // shape[0] = -1;
        return ValueWithOffsets(builder.create<mlir::memref::LoadOp>(
                                    loc, val, std::vector<mlir::Value>({c0})),
                                /*isReference*/ true);
      }
    }
    return ValueWithOffsets(val, /*isReference*/ true);
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

// TODO: too long and difficult to understand.
void ValueWithOffsets::store(mlir::OpBuilder &builder, ValueWithOffsets toStore,
                             bool isArray) const {
  assert(toStore.val);
  if (isArray) {
    if (!toStore.isReference) {
      llvm::errs() << " toStore.val: " << toStore.val << " isref "
                   << toStore.isReference << " isar" << isArray << "\n";
    }
    assert(toStore.isReference);
    auto loc = builder.getUnknownLoc();
    auto zeroIndex = builder.create<mlir::ConstantIndexOp>(loc, 0);

    if (auto smt = toStore.val.getType().dyn_cast<mlir::MemRefType>()) {
      assert(smt.getShape().size() <= 2);

      if (auto mt = val.getType().dyn_cast<mlir::MemRefType>()) {
        assert(smt.getElementType() == mt.getElementType());
        if (mt.getShape().size() != smt.getShape().size()) {
          llvm::errs() << " val: " << val << " tsv: " << toStore.val << "\n";
          llvm::errs() << " mt: " << mt << " smt: " << smt << "\n";
        }
        assert(mt.getShape().size() == smt.getShape().size());
        assert(smt.getShape().back() == mt.getShape().back());

        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<mlir::ConstantIndexOp>(loc, i));
          builder.create<mlir::memref::StoreOp>(
              loc, builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
              val, idx);
        }
      } else {
        auto pt = val.getType().cast<mlir::LLVM::LLVMPointerType>();
        mlir::Type elty;
        if (auto at = pt.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
          elty = at.getElementType();
          assert(smt.getShape().back() == at.getNumElements());
        } else {
          auto st = pt.getElementType().dyn_cast<LLVM::LLVMStructType>();
          elty = st.getBody()[0];
          assert(smt.getShape().back() == (ssize_t)st.getBody().size());
        }
        assert(elty == smt.getElementType());
        elty = LLVM::LLVMPointerType::get(elty, pt.getAddressSpace());

        auto iTy = builder.getIntegerType(32);
        auto zero32 = builder.create<mlir::ConstantOp>(
            loc, iTy, builder.getIntegerAttr(iTy, 0));
        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<mlir::ConstantIndexOp>(loc, i));
          mlir::Value lidx[] = {val, zero32,
                                builder.create<mlir::ConstantOp>(
                                    loc, iTy, builder.getIntegerAttr(iTy, i))};
          builder.create<mlir::LLVM::StoreOp>(
              loc, builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
              builder.create<mlir::LLVM::GEPOp>(loc, elty, lidx));
        }
      }
    } else if (auto smt = val.getType().dyn_cast<mlir::MemRefType>()) {
      assert(smt.getShape().size() <= 2);

      auto pt = toStore.val.getType().cast<LLVM::LLVMPointerType>();
      mlir::Type elty;
      if (auto at = pt.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
        elty = at.getElementType();
        assert(smt.getShape().back() == at.getNumElements());
      } else {
        auto st = pt.getElementType().dyn_cast<LLVM::LLVMStructType>();
        elty = st.getBody()[0];
        assert(smt.getShape().back() == (ssize_t)st.getBody().size());
      }
      assert(elty == smt.getElementType());
      elty = LLVM::LLVMPointerType::get(elty, pt.getAddressSpace());

      auto iTy = builder.getIntegerType(32);
      auto zero32 = builder.create<mlir::ConstantOp>(
          loc, iTy, builder.getIntegerAttr(iTy, 0));
      for (ssize_t i = 0; i < smt.getShape().back(); i++) {
        SmallVector<mlir::Value, 2> idx;
        if (smt.getShape().size() == 2)
          idx.push_back(zeroIndex);
        idx.push_back(builder.create<mlir::ConstantIndexOp>(loc, i));
        mlir::Value lidx[] = {toStore.val, zero32,
                              builder.create<mlir::ConstantOp>(
                                  loc, iTy, builder.getIntegerAttr(iTy, i))};
        builder.create<mlir::memref::StoreOp>(
            loc,
            builder.create<mlir::LLVM::LoadOp>(
                loc, builder.create<mlir::LLVM::GEPOp>(loc, elty, lidx)),
            val, idx);
      }
    } else
      store(builder, toStore.getValue(builder));
  } else {
    store(builder, toStore.getValue(builder));
  }
}
