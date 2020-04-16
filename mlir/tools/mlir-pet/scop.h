#ifndef PETMLIR_PET_SCOP_H
#define PETMLIR_PET_SCOP_H

#include "typeTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "isl/isl-noexceptions.h"

class pet_scop;
class pet_stmt;

namespace pet {

enum class ElementType { FLOAT, DOUBLE, INT };

// wrapper around pet_array.
class PetArray {

public:
  PetArray(std::string name, isl::set context, isl::set extent,
           isl::set valueBounds, ElementType elementType, int elementIsRecord,
           int elementSize, int liveOut, int uniquelyDefined, int declared,
           int exposed, int outer)
      : name_(name), context_(context), extent_(extent),
        valueBounds_(valueBounds), elementType_(elementType),
        elementIsRecord_(elementIsRecord), elementSize_(elementSize),
        liveOut_(liveOut), uniquelyDefined_(uniquelyDefined),
        declared_(declared), exposed_(exposed), outer_(outer) {}

  // dump the current petArray.
  void dump() const;

  // is the array declared somewhere in the scop?
  bool isDeclared() const;

  // get array dimensionality.
  size_t getDimensionality() const;

  // get the extent along the "i" dimension.
  int64_t getExtentOnDimension(size_t dim) const;

  // get the name.
  std::string getName() const;

  // get the name as id.
  isl::id getNameAsId() const;

  // get the type.
  ElementType getType() const;

private:
  // name of the array.
  std::string name_;

  // holds the constraints on the parameter that ensure
  // that this array has a valid size.
  isl::set context_;

  // holds constraints on the indexes.
  isl::set extent_;

  // holds constraints on the element of the array.
  isl::set valueBounds_;

  // array type
  ElementType elementType_;

  // set id the type is a record type
  int elementIsRecord_;

  // is the size in bytes of each array access.
  int elementSize_;

  // is set if the array appears in a live-out pragma.
  int liveOut_;

  // if set then the array is written by a single access.
  int uniquelyDefined_;

  // set if the array was declared somewhere inside the scop.
  int declared_;

  // set if the array declared array is visible outside the scop.
  int exposed_;

  // set if the type of the array is a record and the fields of
  // this record are represented by separate pet_array structures.
  int outer_;
};

template <typename T>
/// A wrapper class around an isl C object, convertible (with copy) to the
/// respective isl C++ type and assignable from such type.
/// This class is inteded to provide access to isl C objects hidden inside
/// other objects without exposing isl C API.
class IslCopyRefWrapper {
public:
  IslCopyRefWrapper(isl_unwrap_t<T> &r) : ref(r) {}

  const T &operator=(const T &rhs) {
    // This will create a C++ wrapper, destroy it immediatly and thus call the
    // appropriate cleaning function for ref.
    isl::manage(ref);

    ref = rhs.copy();
    return rhs;
  }

  operator T() { return isl::manage_copy(ref); }

private:
  isl_unwrap_t<T> &ref;
};

class Scop {
public:
  explicit Scop(pet_scop *scop);
  Scop(const Scop &) = delete;
  Scop(Scop &&) = default;
  static Scop parseFile(isl::ctx ctx, const std::string filename);

  ~Scop();

  // pet_scop does not feature a copy function.
  Scop &operator=(const Scop &) = delete;
  Scop &operator=(Scop &&) = default;

  // dump the current scop.
  void dump() const;

  // obtain the isl context in which Scop lives.
  isl::ctx getCtx() const;

  // return a copy of the schedule.
  isl::schedule getSchedule() const;
  isl::union_map getScheduleAsUnionMap() const;

  // modify schedule.
  IslCopyRefWrapper<isl::schedule> schedule();

  // return a copy of the context.
  isl::set getContext() const;

  // get scop domain.
  isl::union_set getDomain() const;

  isl::union_map getReads() const;
  isl::union_map getMayWrites() const;
  isl::union_map getMustWrites() const;

  // get all deps.
  isl::union_map getAllDependences() const;

  // return the statement associated with "id".
  pet_stmt *getStmt(isl::id id) const;

  // return the tensors that are inputs of
  // the current scop.
  llvm::SmallVector<PetArray, 4> getInputTensors();

  // return the petArray with id "id"
  PetArray getArrayFromId(isl::id id);

private:
  // pointer to pet_scop.
  pet_scop *scop_;

  // all arrays within the scop.
  llvm::SmallVector<PetArray, 4> petArrays_;
};

} // end namespace pet

#endif
