#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::polygeist {
LogicalResult scfParallelUnrollByFactor(
    scf::ParallelOp &pop, uint64_t unrollFactor, unsigned dim,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn);
static LogicalResult scfParallelUnrollByFactors(
    scf::ParallelOp &pop, ArrayRef<uint64_t> unrollFactors,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  unsigned dims = pop.getUpperBound().size();
  assert(dims == unrollFactors.size());
  bool succeeded = true;
  for (unsigned dim = 0; dim < dims; dim++) {
    succeeded = succeeded && polygeist::scfParallelUnrollByFactor(
                                 pop, unrollFactors[dim], dim, annotateFn)
                                 .succeeded();
  }
  return success(succeeded);
}
} // namespace mlir::polygeist
