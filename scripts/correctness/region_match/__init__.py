"""Generic, routine-agnostic region matcher for LoopLibrarian.

    match_regions(mlir_text, spec_dir) -> list[RegionMatch]

Pipeline (see issues/region_match/REPORT.md):
  1. symexec     : tensor/memref IR  -> memory-level affine statement program
                   (tensor SSA re-bufferised; linalg.generic with memref
                   semantics; scratch tensors/allocas privatised)
  2. equations   : statement program -> element equations over input values
                   and final output values, with an affine dependence check
                   on every read (input / final / accumulator, else reject)
  3. egg         : egglog equivalence of equation systems (kernel_match Term +
                   algebra_rules, plus index renaming, layout permutation and
                   triangular-domain <-> mask rules)
  4. matcher     : generic search over operand / parameter / layout bindings
                   against every specification in <spec_dir>/manifest
"""
from .matcher import RegionMatch, load_specs, match_text


def match_regions(mlir_text: str, spec_dir: str) -> list[RegionMatch]:
    return match_text(mlir_text, spec_dir)


__all__ = ["RegionMatch", "match_regions", "load_specs"]
