//===- TypeUtils.cc ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "CodeGenTypes.h"

#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"

#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/identity.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

namespace mlirclang {

using namespace llvm;

bool isRecursiveStruct(Type *T, Type *Meta, SmallPtrSetImpl<Type *> &Seen) {
  if (Seen.count(T))
    return false;
  Seen.insert(T);
  if (T->isVoidTy() || T->isFPOrFPVectorTy() || T->isIntOrIntVectorTy())
    return false;
  if (T == Meta)
    return true;
  for (Type *ST : T->subtypes()) {
    if (isRecursiveStruct(ST, Meta, Seen))
      return true;
  }
  return false;
}

bool isRecursiveStruct(
    clang::QualType QT,
    SmallPtrSetImpl<const clang::Type *> &RecordEncountered) {

  std::optional<bool> isRecursive =
      TypeSwitch<clang::QualType, std::optional<bool>>(QT)
          .Case<clang::ElaboratedType>([&](const auto *ET) {
            return isRecursiveStruct(ET->getNamedType(), RecordEncountered);
          })
          .Case<clang::UsingType>([&](const auto *UT) {
            return isRecursiveStruct(UT->getUnderlyingType(),
                                     RecordEncountered);
          })
          .Case<clang::ParenType>([&](const auto *PT) {
            return isRecursiveStruct(PT->getInnerType(), RecordEncountered);
          })
          .Case<clang::DeducedType>([&](const auto *DT) {
            return isRecursiveStruct(DT->getDeducedType(), RecordEncountered);
          })
          .Case<clang::SubstTemplateTypeParmType>([&](const auto *STT) {
            return isRecursiveStruct(STT->getReplacementType(),
                                     RecordEncountered);
          })
          .Case<clang::TemplateSpecializationType, clang::TypedefType,
                clang::DecltypeType>([&](const auto *T) {
            return isRecursiveStruct(T->desugar(), RecordEncountered);
          })
          .Case<clang::DecayedType>([&](const auto *DT) {
            return isRecursiveStruct(DT->getOriginalType(), RecordEncountered);
          })
          .Case<clang::ComplexType>([&](const auto *CT) {
            return isRecursiveStruct(CT->getElementType(), RecordEncountered);
          })
          .Case<clang::RecordType>([&](const auto *RT) {
            if (RecordEncountered.count(RT))
              return true;

            RecordEncountered.insert(RT);
            auto *CXRD = dyn_cast<clang::CXXRecordDecl>(RT->getDecl());
            if (CXRD && CXRD->hasDefinition())
              for (auto B : CXRD->bases())
                if (isRecursiveStruct(B.getType(), RecordEncountered))
                  return true;

            for (auto *F : RT->getDecl()->fields())
              if (isRecursiveStruct(F->getType(), RecordEncountered))
                return true;

            RecordEncountered.erase(RT);
            return false;
          })
          .Default([](auto) { return std::nullopt; });

  if (isRecursive.has_value()) {
    return *isRecursive;
  }

  const clang::Type *T = QT->getUnqualifiedDesugaredType();

  if (T->isBuiltinType())
    return false;

  return TypeSwitch<const clang::Type *, bool>(T)
      .Case<clang::ArrayType, clang::VectorType>([&](const auto *VAT) {
        return isRecursiveStruct(VAT->getElementType(), RecordEncountered);
      })
      .Case<clang::FunctionProtoType, clang::FunctionNoProtoType,
            clang::EnumType>([&](const auto *T) { return false; })
      .Case<clang::PointerType, clang::ReferenceType>([&](const auto *T) {
        return isRecursiveStruct(T->getPointeeType(), RecordEncountered);
      });
}

Type *anonymize(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    return PT;

  if (auto *AT = dyn_cast<ArrayType>(T))
    return ArrayType::get(anonymize(AT->getElementType()),
                          AT->getNumElements());
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    SmallVector<Type *, 4> V;
    for (auto *T : FT->params())
      V.push_back(anonymize(T));
    return FunctionType::get(anonymize(FT->getReturnType()), V, FT->isVarArg());
  }
  if (auto *ST = dyn_cast<StructType>(T)) {
    if (ST->isLiteral())
      return ST;

    SmallVector<Type *, 4> V;
    for (auto *T : ST->elements()) {
      SmallPtrSet<Type *, 4> Seen;
      if (isRecursiveStruct(T, ST, Seen))
        V.push_back(T);
      else
        V.push_back(anonymize(T));
    }
    return StructType::get(ST->getContext(), V, ST->isPacked());
  }
  return T;
}

mlir::IntegerAttr wrapIntegerMemorySpace(unsigned MemorySpace,
                                         mlir::MLIRContext *Ctx) {
  return MemorySpace ? mlir::IntegerAttr::get(mlir::IntegerType::get(Ctx, 64),
                                              MemorySpace)
                     : nullptr;
}

unsigned getAddressSpace(mlir::Type Ty) {
  return llvm::TypeSwitch<mlir::Type, unsigned>(Ty)
      .Case<mlir::MemRefType>(
          [](auto MemRefTy) { return MemRefTy.getMemorySpaceAsInt(); })
      .Case<mlir::LLVM::LLVMPointerType>(
          [](auto PtrTy) { return PtrTy.getAddressSpace(); });
}

mlir::Type getPtrTyWithNewType(mlir::Type Orig, mlir::Type NewElementType) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(Orig)
      .Case<mlir::MemRefType>([NewElementType](auto Ty) {
        return mlir::MemRefType::get(Ty.getShape(), NewElementType,
                                     Ty.getLayout(), Ty.getMemorySpace());
      })
      .Case<mlir::LLVM::LLVMPointerType>(
          llvm::identity<mlir::LLVM::LLVMPointerType>());
}

template <typename F>
static typename std::invoke_result_t<F, uint32_t>::value_type
symbolizeAttr(const clang::TemplateArgument &templArg, F symbolize) {
  auto optVal =
      symbolize(static_cast<uint32_t>(templArg.getAsIntegral().getZExtValue()));
  assert(optVal && "Invalid enum value");
  return *optVal;
}

llvm::Type *getLLVMType(const clang::QualType QT,
                        clang::CodeGen::CodeGenModule &CGM) {
  if (QT->isVoidType())
    return llvm::Type::getVoidTy(CGM.getModule().getContext());

  return CGM.getTypes().ConvertType(QT);
}

template <typename MLIRTy> static bool isTyOrTyVectorTy(mlir::Type Ty) {
  if (isa<MLIRTy>(Ty))
    return true;
  const auto VecTy = dyn_cast<mlir::VectorType>(Ty);
  return VecTy && isa<MLIRTy>(VecTy.getElementType());
}

bool isFPOrFPVectorTy(mlir::Type Ty) {
  return isTyOrTyVectorTy<mlir::FloatType>(Ty);
}

bool isIntOrIntVectorTy(mlir::Type Ty) {
  return isTyOrTyVectorTy<mlir::IntegerType>(Ty);
}

unsigned getPrimitiveSizeInBits(mlir::Type Ty) {
  return llvm::TypeSwitch<mlir::Type, unsigned>(Ty)
      .Case<mlir::IntegerType, mlir::FloatType>(
          [](auto Ty) { return Ty.getWidth(); })
      .Case<mlir::IndexType>(
          [](auto) { return mlir::IndexType::kInternalStorageBitWidth; })
      .Case<mlir::VectorType>([](auto VecTy) {
        return VecTy.getNumElements() *
               getPrimitiveSizeInBits(VecTy.getElementType());
      });
}

} // namespace mlirclang
