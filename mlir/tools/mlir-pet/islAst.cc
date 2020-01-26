#include "islAst.h"
#include "llvm/Support/raw_ostream.h"

using namespace ast;
using namespace pet;
using namespace llvm;

IslAst::IslAst(Scop &scop) : scop_(scop) {
  auto schedule = scop_.getSchedule();
  isl::ast_build build = isl::ast_build(scop_.getCtx());
  root_ = build.node_from_schedule(schedule);
}

void IslAst::dump() const { outs() << root_.to_str() << "\n"; }

isl::ast_node IslAst::getRoot() const { return root_; }

Scop &IslAst::getScop() const { return scop_; }
