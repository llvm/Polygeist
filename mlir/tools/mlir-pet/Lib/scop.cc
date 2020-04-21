#include "scop.h"
#include "ctx.h"
#include "operator.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace pet;
using namespace llvm;

void PetArray::dump() const {
  outs() << "name: " << name_ << "\n";
  outs() << "context: " << context_.to_str() << "\n";
  outs() << "extent: " << extent_.to_str() << "\n";
  outs() << "value bounds: " << valueBounds_.to_str() << "\n";
  outs() << "element is record: " << elementIsRecord_ << "\n";
  outs() << "element size: " << elementSize_ << "\n";
  outs() << "live out: " << liveOut_ << "\n";
  outs() << "uniquely defined: " << uniquelyDefined_ << "\n";
  outs() << "declared: " << declared_ << "\n";
  outs() << "exposed: " << exposed_ << "\n";
  outs() << "outer: " << outer_ << "\n";
}

bool PetArray::isDeclared() const {
  if (declared_)
    return true;
  return false;
}

std::string PetArray::getName() const { return name_; }

ElementType PetArray::getType() const { return elementType_; }

size_t PetArray::getDimensionality() const {
  auto space = extent_.get_space();
  return space.dim(isl::dim::out);
}

// TODO: likely not a good way to convert isl::val to int64_t.
// TODO: handle in case the extent is parametric.
int64_t PetArray::getExtentOnDimension(size_t dim) const {
  assert((dim <= extent_.get_space().dim(isl::dim::out)) &&
         "dim should be less than the array dimensionality");
  auto pwaffMax = extent_.dim_max(dim);
  auto pwaffMin = extent_.dim_min(dim);

  if ((pwaffMax.n_piece() != 1) || (pwaffMin.n_piece() != 1))
    llvm_unreachable("expect single piece for min and max");

  isl::val valueMax;
  isl::val valueMin;

  pwaffMax.foreach_piece([&valueMax](isl::set s, isl::aff a) -> isl_stat {
    valueMax = a.get_constant_val();
    return isl_stat_ok;
  });
  pwaffMin.foreach_piece([&valueMin](isl::set s, isl::aff a) -> isl_stat {
    valueMin = a.get_constant_val();
    return isl_stat_ok;
  });

  valueMax = valueMax.sub(valueMin);
  valueMax = valueMax.add(isl::val::one(valueMax.get_ctx()));
  return std::stoll(valueMax.to_str());
}

void Scop::dump() const {
  pet_scop_dump(scop_);
  outs() << "Arrays in scop: \n";
  for (const auto petArray : petArrays_) {
    petArray.dump();
  }
}

static ElementType getType(std::string typeAsString) {
  if (typeAsString == "float")
    return ElementType::FLOAT;
  if (typeAsString == "double")
    return ElementType::DOUBLE;
  if (typeAsString == "int")
    return ElementType::INT;
  llvm_unreachable("unknown type");
}

isl::set getExtent(isl::union_map accesses, isl::space arraySpace) {
  auto uSet = accesses.range();
  uSet = uSet.coalesce();
  uSet = uSet.detect_equalities();
  uSet = uSet.coalesce();

  if (uSet.is_empty())
    return isl::set::empty(arraySpace);

  return uSet.extract_set(arraySpace);
}

Scop::Scop(pet_scop *scop) : scop_(scop) {
  if (scop) {
    auto reads = getReads();
    auto writes = getMayWrites();

    size_t size = scop_->n_array;
    for (size_t i = 0; i < size; i++) {
      auto context = isl::manage_copy(scop_->arrays[i]->context);
      auto extent =
          getExtent(reads.unite(writes),
                    isl::manage_copy(scop_->arrays[i]->extent).get_space());
      auto valueBounds = isl::manage_copy(scop_->arrays[i]->value_bounds);
      auto elementType = getType(std::string(scop_->arrays[i]->element_type));
      auto elementIsRecord = scop_->arrays[i]->element_is_record;
      auto elementSize = scop_->arrays[i]->element_size;
      auto liveOut = scop_->arrays[i]->live_out;
      auto uniquelyDefined = scop_->arrays[i]->uniquely_defined;
      auto declared = scop_->arrays[i]->declared;
      auto exposed = scop_->arrays[i]->exposed;
      auto outer = scop_->arrays[i]->outer;
      auto name = std::string(extent.get_tuple_name());
      petArrays_.push_back(PetArray(
          name, context, extent, valueBounds, elementType, elementIsRecord,
          elementSize, liveOut, uniquelyDefined, declared, exposed, outer));
    }
  }
}

Scop::~Scop() { pet_scop_free(scop_); }

Scop Scop::parseFile(isl::ctx ctx, const std::string filename) {
  return Scop(
      pet_scop_extract_from_C_source(ctx.get(), filename.c_str(), nullptr));
}

isl::ctx Scop::getCtx() const {
  return isl::ctx(isl_schedule_get_ctx(scop_->schedule));
}

isl::schedule Scop::getSchedule() const {
  return isl::manage_copy(scop_->schedule);
}

IslCopyRefWrapper<isl::schedule> Scop::schedule() {
  return IslCopyRefWrapper<isl::schedule>(scop_->schedule);
}

isl::union_map Scop::getScheduleAsUnionMap() const {
  return getSchedule().get_map();
}

isl::union_set Scop::getDomain() const {
  auto schedule = getSchedule();
  return schedule.get_domain();
}

isl::union_set Scop::getNonKilledDomain() const {
  isl::union_set domain = isl::union_set::empty(getContext().get_space());
  for (int i = 0; i < scop_->n_stmt; i++) {
    struct pet_stmt *stmt = scop_->stmts[i];
    if (pet_stmt_is_kill(stmt))
      continue;
    auto domainI = isl::manage_copy(stmt->domain);
    domain = domain.add_set(domainI);
  }
  return domain;
}

isl::union_map Scop::getReads() const {
  return isl::manage(pet_scop_get_may_reads(scop_));
}

isl::union_map Scop::getMayWrites() const {
  return isl::manage(pet_scop_get_may_writes(scop_));
}

isl::union_map Scop::getMustWrites() const {
  return isl::manage(pet_scop_get_must_writes(scop_));
}

isl::union_map Scop::getAllDependences() const {
  // For the simplest possible dependence analysis, get rid of reference tags.
  auto reads = getReads().domain_factor_domain();
  auto mayWrites = getMayWrites().domain_factor_domain();
  auto mustWrites = getMustWrites().domain_factor_domain();
  auto schedule = getSchedule();

  // False dependences (output and anti).
  // Sinks are writes, sources are reads and writes.
  auto falseDepsFlow = isl::union_access_info(mayWrites.unite(mustWrites))
                           .set_may_source(mayWrites.unite(reads))
                           .set_must_source(mustWrites)
                           .set_schedule(schedule)
                           .compute_flow();

  isl::union_map falseDeps = falseDepsFlow.get_may_dependence();

  // Flow dependences.
  // Sinks are reads and sources are writes.
  auto flowDepsFlow = isl::union_access_info(reads)
                          .set_may_source(mayWrites)
                          .set_must_source(mustWrites)
                          .set_schedule(schedule)
                          .compute_flow();

  isl::union_map flowDeps = flowDepsFlow.get_may_dependence();

  return flowDeps.unite(falseDeps);
}

isl::set Scop::getContext() const { return isl::manage_copy(scop_->context); }

bool Scop::isValid() const { return scop_ != nullptr; }

static isl::id getStmtId(const pet_stmt *stmt) {
  isl::set domain = isl::manage_copy(stmt->domain);
  return domain.get_tuple_id();
}

pet_stmt *Scop::getStmt(isl::id id) const {
  using namespace operators;
  for (int i = 0; i < scop_->n_stmt; i++) {
    if (getStmtId(scop_->stmts[i]) == id)
      return scop_->stmts[i];
  }
  return nullptr;
}

SmallVector<PetArray, 4> Scop::getInputArrays() {
  SmallVector<PetArray, 4> res;
  for (const auto petArray : petArrays_) {
    if (!petArray.isDeclared())
      res.push_back(petArray);
  }
  return res;
}

PetArray Scop::getArrayFromId(isl::id id) {
  for (auto const petArray : petArrays_) {
    if (petArray.getName() == id.to_str())
      return petArray;
  }
  llvm_unreachable("cannot find array with the provided id");
}
