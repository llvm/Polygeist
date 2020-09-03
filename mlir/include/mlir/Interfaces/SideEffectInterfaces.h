//===- SideEffectInterfaces.h - SideEffect in MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains traits, interfaces, and utilities for defining and
// querying the side effects of an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SIDEEFFECTS_H
#define MLIR_INTERFACES_SIDEEFFECTS_H

#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StorageUniquerSupport.h"

namespace mlir {
namespace SideEffects {

namespace detail {
class EffectUniquer;
} // namespace detail

/// The default storage class for Effects. Only contains the type ID of the
/// effect.
class EffectStorage : public StorageUniquer::BaseStorage {
  friend detail::EffectUniquer;
  friend StorageUniquer;
  friend MLIRContext;

public:
  /// Constructs a storage with the given type ID.
  explicit EffectStorage(const TypeID &id) : typeID(id) {}

  /// Returns type ID of the storage.
  TypeID getTypeID() const { return typeID; }

private:
  /// Friends can construct a storage and initialize it later.
  EffectStorage() {}
  void initialize(const TypeID &id) { typeID = id; }

  /// Type ID of the effect.
  TypeID typeID;
};

namespace detail {
/// A utility class to get or create unique instances of side effects within an
/// MLIRContext.
class EffectUniquer {
public:
  /// Get a uniqued non-parametric side effect.
  // TODO: support parametric side effects
  template <typename T>
  static typename std::enable_if<
      std::is_same<typename T::ImplType, EffectStorage>::value, T>::type
  get(MLIRContext *ctx) {
    return ctx->getEffectUniquer().get<typename T::ImplType>(T::getTypeID());
  }

  template <typename T, typename... Args>
  static typename std::enable_if<
      !std::is_same<typename T::ImplType, EffectStorage>::value, T>::type
  get(MLIRContext *ctx, Args &&...args) {
    return ctx->getEffectUniquer().get<typename T::ImplType>(
        [](EffectStorage *storage) { storage->initialize(T::getTypeID()); },
        T::getTypeID(), std::forward<Args>(args)...);
  }
};
} // namespace detail

//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

/// This class represents a base class for a specific effect type.
class Effect {
public:
  template <typename DerivedEffect, typename BaseEffect = Effect,
            typename StorageType = EffectStorage>
  using Base =
      mlir::detail::StorageUserBase<DerivedEffect, BaseEffect, StorageType,
                                    detail::EffectUniquer>;
  using ImplType = EffectStorage;

  /// Return the unique identifier for the base effects class.
  TypeID getTypeID() const { return impl->getTypeID(); }

  /// Support for LLVM-style casting.
  template <typename U> bool isa() const;
  template <typename First, typename Second, typename... Rest> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  /// Support casting to itself.
  static bool classof(Effect) { return true; }

  Effect() : impl(nullptr) {}
  /*implicit*/ Effect(ImplType *impl) : impl(impl) {}
  Effect(const Effect &) = default;

protected:
  ImplType *impl;
};

template <typename U> bool Effect::isa() const {
  assert(impl && "isa<> used on a null attribute.");
  return U::classof(*this);
}

template <typename First, typename Second, typename... Rest>
bool Effect::isa() const {
  return isa<First>() || isa<Second, Rest...>();
}

template <typename U> U Effect::dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);
}
template <typename U> U Effect::dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);
}
template <typename U> U Effect::cast() const {
  assert(isa<U>());
  return U(impl);
}

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

/// This class represents a specific resource that an effect applies to. This
/// class represents an abstract interface for a given resource.
class Resource {
public:
  virtual ~Resource() {}

  /// This base class is used for derived effects that are non-parametric.
  template <typename DerivedResource, typename BaseResource = Resource>
  class Base : public BaseResource {
  public:
    using BaseT = Base<DerivedResource>;

    /// Returns a unique instance for the given effect class.
    static DerivedResource *get() {
      static DerivedResource instance;
      return &instance;
    }

    /// Return the unique identifier for the base resource class.
    static TypeID getResourceID() { return TypeID::get<DerivedResource>(); }

    /// 'classof' used to support llvm style cast functionality.
    static bool classof(const Resource *resource) {
      return resource->getResourceID() == BaseT::getResourceID();
    }

  protected:
    Base() : BaseResource(BaseT::getResourceID()){};
  };

  /// Return the unique identifier for the base resource class.
  TypeID getResourceID() const { return id; }

  /// Return a string name of the resource.
  virtual StringRef getName() = 0;

protected:
  Resource(TypeID id) : id(id) {}

private:
  /// The id of the derived resource class.
  TypeID id;
};

/// A conservative default resource kind.
struct DefaultResource : public Resource::Base<DefaultResource> {
  StringRef getName() final { return "<Default>"; }
};

/// An automatic allocation-scope resource that is valid in the context of a
/// parent AutomaticAllocationScope trait.
struct AutomaticAllocationScopeResource
    : public Resource::Base<AutomaticAllocationScopeResource> {
  StringRef getName() final { return "AutomaticAllocationScope"; }
};

/// This class represents a specific instance of an effect. It contains the
/// effect being applied, a resource that corresponds to where the effect is
/// applied, and an optional value(either operand, result, or region entry
/// argument) that the effect is applied to.
template <typename EffectT> class EffectInstance {
public:
  EffectInstance(EffectT effect, Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource) {}
  EffectInstance(EffectT effect, Value value,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value) {}

  /// Return the effect being applied.
  EffectT getEffect() const { return effect; }

  /// Return the value the effect is applied on, or nullptr if there isn't a
  /// known value being affected.
  Value getValue() const { return value; }

  /// Return the resource that the effect applies to.
  Resource *getResource() const { return resource; }

private:
  /// The specific effect being applied.
  EffectT effect;

  /// The resource that the given value resides in.
  Resource *resource;

  /// The value that the effect applies to. This is optionally null.
  Value value;
};
} // namespace SideEffects

//===----------------------------------------------------------------------===//
// SideEffect Traits
//===----------------------------------------------------------------------===//

namespace OpTrait {
/// This trait indicates that the side effects of an operation includes the
/// effects of operations nested within its regions. If the operation has no
/// derived effects interfaces, the operation itself can be assumed to have no
/// side effects.
template <typename ConcreteType>
class HasRecursiveSideEffects
    : public TraitBase<ConcreteType, HasRecursiveSideEffects> {};
} // namespace OpTrait

//===----------------------------------------------------------------------===//
// Operation Memory-Effect Modeling
//===----------------------------------------------------------------------===//

namespace MemoryEffects {
/// This class represents the base class used for memory effects.
struct Effect : public SideEffects::Effect {
  using SideEffects::Effect::Effect;

  /// A base class for memory effects that provides helper utilities.
  template <typename DerivedEffect>
  using Base = SideEffects::Effect::Base<DerivedEffect, Effect>;

  static bool classof(const SideEffects::Effect *effect);
};
using EffectInstance = SideEffects::EffectInstance<Effect>;

/// The following effect indicates that the operation allocates from some
/// resource. An 'allocate' effect implies only allocation of the resource, and
/// not any visible mutation or dereference.
struct Allocate : public Effect::Base<Allocate> {
  using Base::Base;
};

/// The following effect indicates that the operation frees some resource that
/// has been allocated. An 'allocate' effect implies only de-allocation of the
/// resource, and not any visible allocation, mutation or dereference.
struct Free : public Effect::Base<Free> {
  using Base::Base;
};

/// The following effect indicates that the operation reads from some resource.
/// A 'read' effect implies only dereferencing of the resource, and not any
/// visible mutation.
struct Read : public Effect::Base<Read> {
  using Base::Base;
};

/// The following effect indicates that the operation writes to some resource. A
/// 'write' effect implies only mutating a resource, and not any visible
/// dereference or read.
struct Write : public Effect::Base<Write> {
  using Base::Base;
};
} // namespace MemoryEffects

//===----------------------------------------------------------------------===//
// Affine Memory Effects
//===----------------------------------------------------------------------===//

namespace AffineMemoryEffects {
/// Storage class for integer-set based side effects.
// This is similar to type/attribute storage and everything that isn't already
// owned by the context should be copied into the allocator.
class IntegerSetEffectStorage : public SideEffects::EffectStorage {
public:
  explicit IntegerSetEffectStorage(IntegerSet indices)
      : EffectStorage(TypeID::get<IntegerSetEffectStorage>()),
        affectedIndices(indices) {}

  using KeyTy = IntegerSet;

  bool operator==(const KeyTy &other) const { return other == affectedIndices; }

  static IntegerSetEffectStorage *
  construct(StorageUniquer::StorageAllocator &allocator, IntegerSet indices) {
    return new (allocator.allocate<IntegerSetEffectStorage>())
        IntegerSetEffectStorage(indices);
  }

  IntegerSet getAffectedIndices() const { return affectedIndices; }

private:
  IntegerSet affectedIndices;
};

/// Base class for side effects that are associated with an integer set.
struct Effect : public SideEffects::Effect {
  using SideEffects::Effect::Effect;

  /// Individual effects must derive this class.
  template <typename DerivedEffect>
  class IntegerSetEffectBase
      : public SideEffects::Effect::Base<DerivedEffect, Effect,
                                         IntegerSetEffectStorage> {
  public:
    using Base<DerivedEffect, Effect, IntegerSetEffectStorage>::Base;
    IntegerSet getAffectedIndices() {
      return this->getImpl()->getAffectedIndices();
    }

    static TypeID getTypeID() { return TypeID::get<DerivedEffect>(); }
  };
};

using EffectInstance = SideEffects::EffectInstance<Effect>;

/// "Read" side effect associated with an integer set.
struct Read : public Effect::IntegerSetEffectBase<Read> {
  using IntegerSetEffectBase::IntegerSetEffectBase;
};

/// "Write" side effect associated with an integer set.
struct Write : public Effect::IntegerSetEffectBase<Write> {
  using IntegerSetEffectBase::IntegerSetEffectBase;
};

} // namespace AffineMemoryEffects

//===----------------------------------------------------------------------===//
// SideEffect Utilities
//===----------------------------------------------------------------------===//

/// Return true if the given operation is unused, and has no side effects on
/// memory that prevent erasing.
bool isOpTriviallyDead(Operation *op);

/// Return true if the given operation would be dead if unused, and has no side
/// effects on memory that would prevent erasing. This is equivalent to checking
/// `isOpTriviallyDead` if `op` was unused.
bool wouldOpBeTriviallyDead(Operation *op);

} // end namespace mlir

//===----------------------------------------------------------------------===//
// SideEffect Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the side effect interfaces.
#include "mlir/Interfaces/SideEffectInterfaces.h.inc"

#endif // MLIR_INTERFACES_SIDEEFFECTS_H
