#!/usr/bin/env python3
"""Structured equality saturation for mixed loop/Linalg regions.

This module deliberately separates two responsibilities:

* MLIR proves imperative legality (nesting, single-writer temporaries and
  analyzable affine/scf loops).
* Egglog proves scheduling equivalences over the resulting pure region while
  reusing ``kernel_match.Term`` for scalar algebra.

It is intentionally conservative: unknown side effects, ambiguous memref
writers, or non-structured control flow form region boundaries rather than
being guessed through.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from egglog import EGraph, Expr, StringLike, rewrite, ruleset, vars_

from kernel_match import (
    GenericBody,
    Term,
    _ast_to_term,
    _parse_term,
    _term_repr,
    algebra_rules,
)


class LoopDomain(Expr):
    """Symbolic iteration domain; bounds remain exact MLIR text."""

    def __init__(self, text: StringLike) -> None: ...

    @classmethod
    def Nest(cls, outer: LoopDomain, inner: LoopDomain) -> LoopDomain: ...


class StructuredComputation(Expr):
    """Pure scheduling view of a legal mixed loop/Linalg region."""

    def __init__(self, name: StringLike) -> None: ...

    @classmethod
    def Map(cls, domain: LoopDomain, body: Term) -> StructuredComputation: ...

    @classmethod
    def Reduce(cls, parallel: LoopDomain, reduction: LoopDomain,
               body: Term) -> StructuredComputation: ...

    @classmethod
    def For(cls, domain: LoopDomain,
            body: StructuredComputation) -> StructuredComputation: ...

    @classmethod
    def Sequence(cls, first: StructuredComputation,
                 second: StructuredComputation) -> StructuredComputation: ...

    @classmethod
    def Fused2(cls, domain: LoopDomain, first: Term,
               second: Term) -> StructuredComputation: ...

    @classmethod
    def Fused3(cls, domain: LoopDomain, first: Term, second: Term,
               third: Term) -> StructuredComputation: ...


outer, inner = vars_("outer inner", LoopDomain)
(reduction_domain,) = vars_("reduction_domain", LoopDomain)
first, second, third = vars_("first second third", Term)
left_comp, right_comp = vars_("left_comp right_comp", StructuredComputation)


def structured_rules():
    """Equivalences enabled only after the encoder establishes legality."""
    return ruleset(
        # A loop around a pure sequence is the same schedule as applying the
        # loop to each stage.  This exposes the two maps to the fusion rules.
        rewrite(StructuredComputation.For(
            outer, StructuredComputation.Sequence(
                left_comp, right_comp))).to(
                    StructuredComputation.Sequence(
                        StructuredComputation.For(outer, left_comp),
                        StructuredComputation.For(outer, right_comp))),
        # Lift a structured parallel operation through an analyzable loop.
        rewrite(StructuredComputation.For(
            outer, StructuredComputation.Map(inner, first))).to(
                StructuredComputation.Map(
                    LoopDomain.Nest(outer, inner), first)),
        rewrite(StructuredComputation.For(
            outer, StructuredComputation.Reduce(
                inner, reduction_domain, first))).to(
                    StructuredComputation.Reduce(
                        LoopDomain.Nest(outer, inner),
                        reduction_domain, first)),
        rewrite(StructuredComputation.For(
            outer, StructuredComputation.Fused2(inner, first, second))).to(
                StructuredComputation.Fused2(
                    LoopDomain.Nest(outer, inner), first, second)),
        rewrite(StructuredComputation.For(
            outer, StructuredComputation.Fused3(
                inner, first, second, third))).to(
                    StructuredComputation.Fused3(
                        LoopDomain.Nest(outer, inner), first, second, third)),
        # Producer-consumer fusion. The encoder constructs Sequence only for
        # a same-domain, pure, single-writer dependence chain.
        rewrite(StructuredComputation.Sequence(
            StructuredComputation.Map(inner, first),
            StructuredComputation.Map(inner, second))).to(
                StructuredComputation.Fused2(inner, first, second)),
        rewrite(StructuredComputation.Sequence(
            StructuredComputation.Fused2(inner, first, second),
            StructuredComputation.Map(inner, third))).to(
                StructuredComputation.Fused3(
                    inner, first, second, third)),
        rewrite(StructuredComputation.Sequence(
            StructuredComputation.Map(inner, first),
            StructuredComputation.Fused2(inner, second, third))).to(
                StructuredComputation.Fused3(
                    inner, first, second, third)),
    )


@dataclass(frozen=True)
class LoopInfo:
    kind: str
    induction: str
    bounds: str
    span: tuple[int, int]

    @property
    def domain_text(self) -> str:
        return f"{self.kind}:{self.induction}:{self.bounds}"


@dataclass(frozen=True)
class ViewInfo:
    result: str
    base: str
    map_text: str
    operands: tuple[str, ...]


@dataclass
class StructuredOp:
    index: int
    span: tuple[int, int]
    loops: tuple[LoopInfo, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_roots: tuple[str, ...]
    output_roots: tuple[str, ...]
    accesses: tuple[str, ...]
    body: GenericBody
    term: Term
    pure: bool

    @property
    def parallel_count(self) -> int:
        return len(self.loops) + sum(
            it == "parallel" for it in self.body.iterator_types)

    @property
    def reduction_count(self) -> int:
        return sum(it == "reduction" for it in self.body.iterator_types)


@dataclass(frozen=True)
class SafetyFacts:
    same_loop_nest: bool
    pure: bool
    single_writer_temporaries: bool
    no_unknown_alias: bool

    @property
    def fusible(self) -> bool:
        return (self.same_loop_nest and self.pure and
                self.single_writer_temporaries and self.no_unknown_alias)


@dataclass
class StructuredRegion:
    operations: list[StructuredOp]
    facts: SafetyFacts
    dependence_roots: tuple[str, ...] = ()


@dataclass
class SaturationResult:
    region: StructuredRegion
    source: StructuredComputation
    lifted: StructuredComputation | None
    fused: StructuredComputation | None
    elapsed_ms: float
    egraph_nodes: int
    iterations: int
    timed_out: bool = False
    extracted_kind: str | None = None
    lowering_blocker: str | None = None
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResidualIdiomCandidate:
    """An idiom recovered from a loop that has not raised to Linalg.

    These records are deliberately analysis-only.  In particular, an
    indirect scatter or CSR traversal is not executable merely because its
    loop shape was recovered: collision semantics, bounds and the concrete
    GPU ABI still have to be validated by a lowering adapter.
    """

    kind: str
    loop: LoopInfo
    evidence: tuple[str, ...]
    lowering_blocker: str


_LOOP_HEADER_RE = re.compile(
    r"\b(affine\.for|scf\.for)\s+(%[\w.$-]+)\s*=\s*([^{}]+?)\{")
_SUBMAP_RE = re.compile(
    r"^\s*(%[\w.$-]+)\s*=\s*polygeist\.submap\s*"
    r"\(\s*(%[\w.$-]+)\s*(?:,([^)]*))?\)\s*"
    r"\{\s*map\s*=\s*([^}]+)\}", re.MULTILINE)
_SSA_RE = re.compile(r"%[\w.$-]+")


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for pos in range(opening, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos + 1
    return None


def parse_loops(text: str) -> list[LoopInfo]:
    loops: list[LoopInfo] = []
    for match in _LOOP_HEADER_RE.finditer(text):
        opening = text.find("{", match.start(), match.end())
        closing = _matching_brace(text, opening)
        if closing is None:
            continue
        loops.append(LoopInfo(
            kind=match.group(1), induction=match.group(2),
            bounds=" ".join(match.group(3).split()),
            span=(match.start(), closing)))
    return loops


def parse_views(text: str) -> dict[str, ViewInfo]:
    views: dict[str, ViewInfo] = {}
    for match in _SUBMAP_RE.finditer(text):
        operands = tuple(_SSA_RE.findall(match.group(3) or ""))
        views[match.group(1)] = ViewInfo(
            result=match.group(1), base=match.group(2),
            map_text=" ".join(match.group(4).split()), operands=operands)
    return views


def resolve_view_root(value: str, views: dict[str, ViewInfo]) -> str:
    seen: set[str] = set()
    while value in views and value not in seen:
        seen.add(value)
        value = views[value].base
    return value


def _ssa_operands(part: str) -> tuple[str, ...]:
    values = part.split(":", 1)[0]
    return tuple(_SSA_RE.findall(values))


def _body_is_pure(body: GenericBody) -> bool:
    allowed = ("arith.", "math.", "linalg.index", "tensor.extract")
    for line in body.body_lines:
        stripped = line.strip()
        if not stripped:
            continue
        operation = stripped.split("=", 1)[-1].strip()
        if not operation.startswith(allowed):
            return False
    return True


def build_structured_ops(text: str, instances: Sequence,
                         bodies: Sequence[GenericBody],
                         terms: Sequence[Term | None]) -> list[StructuredOp]:
    loops = parse_loops(text)
    affine_aliases = {
        name: literal for name, literal in re.findall(
            r"^\s*(#[A-Za-z_]\w*)\s*=\s*"
            r"(affine_map<\([^)]*\)(?:\[[^]]*\])?\s*->\s*\([^)]*\)>)",
            text, re.MULTILINE)
    }
    result: list[StructuredOp] = []
    for index, (instance, body, term) in enumerate(
            zip(instances, bodies, terms)):
        if term is None:
            continue
        # SSA names are scoped per function and are routinely reused.  Parsing
        # only the prefix makes each dictionary entry the nearest dominating
        # submap definition instead of an unrelated same-named value later in
        # the module.
        views = parse_views(text[:instance.span[0]])
        contained = tuple(sorted(
            (loop for loop in loops
             if loop.span[0] < instance.span[0] and
             instance.span[1] < loop.span[1]),
            key=lambda loop: loop.span[0]))
        inputs = _ssa_operands(instance.ins_part)
        outputs = _ssa_operands(instance.outs_part)
        values = inputs + outputs
        accesses: list[str] = []
        for operand_index, value in enumerate(values):
            view = views.get(value)
            root = resolve_view_root(value, views)
            submap = (affine_aliases.get(view.map_text, view.map_text)
                      if view else "direct")
            symbols = ",".join(view.operands) if view else ""
            generic_map = (
                body.indexing_maps[operand_index]
                if operand_index < len(body.indexing_maps) else "unknown")
            accesses.append(
                f"root={root};submap={submap};symbols={symbols};"
                f"generic={generic_map}")
        result.append(StructuredOp(
            index=index, span=instance.span, loops=contained,
            inputs=inputs, outputs=outputs,
            input_roots=tuple(resolve_view_root(v, views) for v in inputs),
            output_roots=tuple(resolve_view_root(v, views) for v in outputs),
            accesses=tuple(accesses),
            body=body, term=term, pure=_body_is_pure(body)))
    return result


def _same_loop_nest(ops: Sequence[StructuredOp]) -> bool:
    return bool(ops) and all(op.loops == ops[0].loops for op in ops)


def _dependence_roots(ops: Sequence[StructuredOp]) -> tuple[str, ...]:
    roots: list[str] = []
    for producer_index, producer in enumerate(ops):
        for consumer in ops[producer_index + 1:]:
            shared = set(producer.output_roots) & set(consumer.input_roots)
            roots.extend(sorted(shared))
    return tuple(dict.fromkeys(roots))


def _safety_facts(ops: Sequence[StructuredOp], text: str) -> SafetyFacts:
    deps = _dependence_roots(ops)
    writer_counts = {
        root: sum(root in op.output_roots for op in ops) for root in deps
    }
    single_writer = all(count == 1 for count in writer_counts.values())
    # Distinct SSA roots are a conservative no-alias proof only for local
    # allocas. Function arguments may alias and are not asserted independent.
    local_allocas = set(re.findall(
        r"^\s*(%[\w.$-]+)\s*=\s*memref\.alloca\b", text, re.MULTILINE))
    no_unknown_alias = all(root in local_allocas for root in deps)
    loop_context_pure = True
    if ops and ops[0].loops:
        outermost = ops[0].loops[0]
        loop_text = text[outermost.span[0]:outermost.span[1]]
        # Scalar address arithmetic and submap construction are represented by
        # the access descriptors.  Any other imperative effect in the parent
        # loop prevents schedule lifting until dependence analysis models it.
        forbidden = re.compile(
            r"\b(?:func\.call|memref\.(?:store|copy|alloc|dealloc)|"
            r"affine\.store|scf\.(?:if|while)|gpu\.)\b|\bcall\s+@")
        loop_context_pure = not forbidden.search(loop_text)
        loop_context_pure = loop_context_pure and all(
            "iter_args" not in loop.bounds for loop in ops[0].loops)
    return SafetyFacts(
        same_loop_nest=_same_loop_nest(ops),
        pure=all(op.pure for op in ops) and loop_context_pure,
        single_writer_temporaries=single_writer,
        no_unknown_alias=no_unknown_alias)


def discover_regions(text: str, instances: Sequence,
                     bodies: Sequence[GenericBody],
                     terms: Sequence[Term | None]) -> list[StructuredRegion]:
    """Find connected producer/consumer DAGs within each common loop nest."""
    ops = build_structured_ops(text, instances, bodies, terms)
    regions: list[StructuredRegion] = []

    # Split first by identical enclosing loops, then retain connected
    # components.  This handles two independent producers both consumed by a
    # later stencil stage (the NPB-MG residual shape).
    runs: list[list[StructuredOp]] = []
    for op in ops:
        if runs and runs[-1][0].loops == op.loops:
            runs[-1].append(op)
        else:
            runs.append([op])
    for run in runs:
        parent = list(range(len(run)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for producer_index, producer in enumerate(run):
            produced = set(producer.output_roots)
            for consumer_index in range(producer_index + 1, len(run)):
                if produced & set(run[consumer_index].input_roots):
                    union(producer_index, consumer_index)
        components: dict[int, list[StructuredOp]] = {}
        for index, op in enumerate(run):
            components.setdefault(find(index), []).append(op)
        for component in components.values():
            if len(component) < 2:
                continue
            facts = _safety_facts(component, text)
            regions.append(StructuredRegion(
                component, facts, _dependence_roots(component)))
        # A reduction generic under one or more ordinary loops is itself a
        # mixed structured region.  Lifting those parent dimensions is the
        # loop-of-GEMV -> GEMM case; it needs no producer temporary.
        connected_indices = {
            op.index for component in components.values()
            if len(component) > 1 for op in component
        }
        for op in run:
            if (op.index not in connected_indices and
                    (op.reduction_count or _classify_single_op(op) is not None)):
                facts = _safety_facts([op], text)
                regions.append(StructuredRegion([op], facts, ()))
    return regions


def _term_ast(op: StructuredOp):
    return _parse_term(_term_repr(op.term))


def _ast_contains(node, kind: str, value=None) -> bool:
    if not isinstance(node, tuple):
        return False
    if node and node[0] == kind and (value is None or
                                     (len(node) > 1 and node[1] == value)):
        return True
    return any(_ast_contains(child, kind, value) for child in node[1:]
               if isinstance(child, tuple))


def _classify_reduction(op: StructuredOp) -> str | None:
    if not op.reduction_count:
        return None
    ast = _term_ast(op)
    # Memref-form linalg.generic parsing numbers the output block arguments
    # after all inputs, whereas tensor form uses Out(0). Accept both encodings.
    accumulator = (_ast_contains(ast, "Out", 0) or
                   _ast_contains(ast, "In", len(op.inputs)))
    if not accumulator:
        return None
    parallel = op.parallel_count
    if _ast_contains(ast, "Select") and _ast_contains(ast, "Cmp"):
        return "scalar_minmax_reduction" if parallel == 0 else "axis_minmax_reduction"
    if isinstance(ast, tuple) and ast[0] == "Mul":
        return "scalar_product_reduction" if parallel == 0 else "axis_product_reduction"
    if _ast_contains(ast, "Add"):
        return "scalar_sum_reduction" if parallel == 0 else "axis_sum_reduction"
    return "structured_reduction"


def _is_gemm_access_shape(op: StructuredOp) -> bool:
    """Recognize A[i,k], B[k,j], C[i,j] on a three-iterator generic."""
    if len(op.body.indexing_maps) != 3:
        return False
    parsed: list[tuple[list[str], list[str]]] = []
    for map_text in op.body.indexing_maps:
        match = re.fullmatch(
            r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>", map_text.strip())
        if not match:
            return False
        dims = [item.strip() for item in match.group(1).split(",") if item.strip()]
        results = [item.strip() for item in match.group(2).split(",") if item.strip()]
        parsed.append((dims, results))
    dims = parsed[0][0]
    if len(dims) != 3 or any(entry[0] != dims for entry in parsed):
        return False
    parallel = [dims[index] for index, kind in enumerate(op.body.iterator_types)
                if kind == "parallel"]
    reductions = [dims[index] for index, kind in enumerate(op.body.iterator_types)
                  if kind == "reduction"]
    if len(parallel) != 2 or len(reductions) != 1:
        return False
    i, j = parallel
    k = reductions[0]
    maps = [set(entry[1]) for entry in parsed]
    return maps[2] == {i, j} and {frozenset(maps[0]), frozenset(maps[1])} == {
        frozenset((i, k)), frozenset((k, j))}


def _classify_stencil(op: StructuredOp) -> str | None:
    if op.reduction_count or len(op.inputs) < 3:
        return None
    # Raised stencils commonly present each neighbour as a submap of one
    # physical input.  Requiring three distinct access descriptors prevents a
    # pointwise expression with repeated scalar operands from being promoted.
    root_counts = {root: op.input_roots.count(root) for root in op.input_roots}
    repeated_roots = {root for root, count in root_counts.items() if count >= 3}
    if not repeated_roots:
        return None
    neighbor_accesses = {
        access for root, access in zip(op.input_roots, op.accesses)
        if root in repeated_roots
    }
    if len(neighbor_accesses) < 3:
        return None
    return "affine_stencil"


def _classify_single_op(op: StructuredOp) -> str | None:
    reduction = _classify_reduction(op)
    if reduction:
        ast = _term_ast(op)
        if (_is_gemm_access_shape(op) and _ast_contains(ast, "In", 0) and
                _ast_contains(ast, "In", 1) and _ast_contains(ast, "Mul") and
                _ast_contains(ast, "Add") and
                (_ast_contains(ast, "Out", 0) or
                 _ast_contains(ast, "In", len(op.inputs)))):
            return "dense_gemm"
        return reduction
    return _classify_stencil(op)


def _gemm_body_equivalent(egraph: EGraph, op: StructuredOp) -> bool:
    """Prove accumulator + lhs*rhs independent of scalar expression order."""
    if len(op.inputs) != 2 or len(op.outputs) != 1:
        return False
    lhs = Term.Access(op.input_roots[0], op.accesses[0])
    rhs = Term.Access(op.input_roots[1], op.accesses[1])
    out_index = len(op.inputs)
    accumulator = Term.Access(op.output_roots[0], op.accesses[out_index])
    canonical = accumulator + lhs * rhs
    try:
        egraph.check(_term_with_accesses(op) == canonical)
        return True
    except Exception:
        return False


def _blas_subtract_body_equivalent(egraph: EGraph, op: StructuredOp) -> bool:
    """Prove `output - lhs*rhs` for a GEMM/GEMV-shaped generic."""
    if len(op.inputs) != 2 or len(op.outputs) != 1:
        return False
    lhs = Term.Access(op.input_roots[0], op.accesses[0])
    rhs = Term.Access(op.input_roots[1], op.accesses[1])
    accumulator = Term.Access(op.output_roots[0], op.accesses[2])
    canonical = accumulator - lhs * rhs
    try:
        egraph.check(_term_with_accesses(op) == canonical)
        return True
    except Exception:
        return False


def _loop_body(text: str, loop: LoopInfo) -> str:
    return text[loop.span[0]:loop.span[1]]


def _spmv_bound_structure(text: str, loop: LoopInfo,
                          all_loops: Sequence[LoopInfo]) -> tuple[str, str] | None:
    """Recover CSR row-pointer or JDS per-row-count traversal bounds."""
    bound_match = re.match(
        r"(%[\w.$-]+)\s+to\s+(%[\w.$-]+)(?:\s+step\s+%[\w.$-]+)?", loop.bounds)
    if not bound_match:
        return None
    lower, upper = bound_match.groups()
    parents = [candidate for candidate in all_loops
               if candidate.span[0] < loop.span[0] and
               loop.span[1] < candidate.span[1]]
    context_start = max((candidate.span[0] for candidate in parents),
                        default=max(0, text.rfind("func.func", 0, loop.span[0])))
    prefix = text[context_start:loop.span[0]]

    casts = {
        result: operand for result, operand in re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.(?:index_cast|ext[su]i|trunci)\s+"
            r"(%[\w.$-]+)", prefix)
    }

    def loaded_buffer(value: str) -> str | None:
        seen: set[str] = set()
        while value in casts and value not in seen:
            seen.add(value)
            value = casts[value]
        matches = re.findall(
            rf"{re.escape(value)}\s*=\s*(?:memref|affine)\.load\s+"
            rf"(%[\w.$-]+)\[[^]]+\]", prefix)
        return matches[-1] if matches else None

    lower_buffer = loaded_buffer(lower)
    upper_buffer = loaded_buffer(upper)
    if lower_buffer and lower_buffer == upper_buffer:
        return "csr_spmv", lower_buffer
    if re.fullmatch(r"%c0(?:_[\w.$-]+)?", lower) and upper_buffer:
        return "jds_spmv", upper_buffer
    return None


def analyze_residual_loops(text: str) -> list[ResidualIdiomCandidate]:
    """Recover indirect idiom candidates which Linalg cannot yet represent.

    The checks intentionally demand dataflow evidence, not operation counts.
    Nested candidates are de-duplicated in favour of the smallest loop that
    contains all required evidence.
    """
    all_loops = parse_loops(text)
    loops = sorted(all_loops, key=lambda loop: loop.span[1] - loop.span[0])
    results: list[ResidualIdiomCandidate] = []
    covered: list[tuple[int, int, str]] = []
    for loop in loops:
        body = _loop_body(text, loop)
        # Histogram: read a bin at a data-dependent index, combine a value,
        # and write the same aggregate back. Tensor insert or memref.store are
        # both accepted, but a constant/induction-only index is not enough.
        has_insert = "tensor.insert " in body or re.search(r"\bmemref\.store\b", body)
        has_read = "tensor.extract " in body or re.search(r"\bmemref\.load\b", body)
        has_combine = bool(re.search(r"\barith\.(?:add[fi]|max[fs]i|min[fs]i)\b", body))
        extract_defs = set(re.findall(
            r"(%[\w.$-]+)\s*=\s*(?:tensor\.extract|memref\.load)\b", body))
        casts = re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.(fptosi|index_cast)\s+(%[\w.$-]+)", body)
        derived = set(extract_defs)
        changed = True
        while changed:
            changed = False
            for result, _, operand in casts:
                if operand in derived and result not in derived:
                    derived.add(result)
                    changed = True
        dynamic_index = any(
            value in derived and re.search(rf"\[{re.escape(value)}(?:,|\])", body)
            for value, _, _ in casts)
        # An arbitrary indirect read/modify/write is only a scatter.  A
        # histogram additionally has binning evidence: either a numeric sample
        # is converted to an integer bin, or the selected bin is incremented
        # by exactly one.  The latter covers integer-bin Parboil histo/tpacf.
        has_numeric_binning = any(
            kind == "fptosi" and operand in derived
            for _, kind, operand in casts)
        constant_ones = set(re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.constant\s+1(?:\.0+)?\b", body))
        # cgeist's stable names retain the literal even if the defining
        # constant lies just outside the candidate loop.
        constant_ones.update(re.findall(r"(%c1(?:_[\w.$-]+)?)\b", body))
        loads = re.findall(
            r"(%[\w.$-]+)\s*=\s*(?:tensor\.extract|(?:memref|affine)\.load)\s+"
            r"(%[\w.$-]+)\[([^]]+)\]", body)
        adds = re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.add[fi]\s+"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)", body)
        stores = re.findall(
            r"(?:tensor\.insert\s+(%[\w.$-]+)\s+into|"
            r"(?:memref|affine)\.store\s+(%[\w.$-]+),)\s*"
            r"(%[\w.$-]+)\[([^]]+)\]", body)
        selects = {
            result: (truthy, falsy)
            for result, truthy, falsy in re.findall(
                r"(%[\w.$-]+)\s*=\s*arith\.select\s+%[\w.$-]+,\s*"
                r"(%[\w.$-]+),\s*(%[\w.$-]+)", body)
        }
        unit_updates = set()
        for result, lhs, rhs in adds:
            if lhs in constant_ones or rhs in constant_ones:
                unit_updates.add(result)
        # Saturating histograms select between old and old+1.
        for result, operands in selects.items():
            if any(operand in unit_updates for operand in operands):
                unit_updates.add(result)
        has_counting_bin = False
        for first_value, second_value, store_buffer, store_index in stores:
            stored = first_value or second_value
            if stored not in unit_updates:
                continue
            if any(load_buffer == store_buffer and load_index.strip() == store_index.strip()
                   for _, load_buffer, load_index in loads):
                has_counting_bin = True
                break
        if (has_insert and has_read and has_combine and dynamic_index and
                (has_numeric_binning or has_counting_bin)):
            if any(start >= loop.span[0] and end <= loop.span[1] and
                   kind == "indirect_histogram" for start, end, kind in covered):
                continue
            results.append(ResidualIdiomCandidate(
                "indirect_histogram", loop,
                ("data-dependent bin index", "read-combine-write update"),
                "needs bin-range proof and atomic/collision-safe GPU lowering"))
            covered.append((loop.span[0], loop.span[1], "indirect_histogram"))
            continue

        # CSR SpMV: an inner loop has symbolic bounds loaded from rowPtr, a
        # second index array supplies the gather into x, and the product is
        # accumulated.  SSA names are intentionally unrestricted.
        load_defs = {
            result: (buffer, indices)
            for result, buffer, indices in re.findall(
                r"(%[\w.$-]+)\s*=\s*memref\.load\s+"
                r"(%[\w.$-]+)\[([^]]+)\]", body)
        }
        bound_structure = _spmv_bound_structure(text, loop, all_loops)
        cast_sources = {
            result: operand for result, operand in re.findall(
                r"(%[\w.$-]+)\s*=\s*arith\.(?:index_cast|ext[su]i|trunci)\s+"
                r"(%[\w.$-]+)", body)
        }
        gather_indices = set(load_defs)
        changed = True
        while changed:
            changed = False
            for result, operand in cast_sources.items():
                if operand in gather_indices and result not in gather_indices:
                    gather_indices.add(result)
                    changed = True
        indirect_gather = False
        for loaded_name in gather_indices:
            if re.search(
                    rf"(?:memref|affine)\.load\s+%[\w.$-]+"
                    rf"\[{re.escape(loaded_name)}\]", body):
                indirect_gather = True
                break
        product_accumulate = (bool(re.search(r"\barith\.mulf\b", body)) and
                              bool(re.search(r"\barith\.addf\b", body)))
        if bound_structure and indirect_gather and product_accumulate:
            spmv_kind, bounds_buffer = bound_structure
            if any(start >= loop.span[0] and end <= loop.span[1] and
                   kind == spmv_kind for start, end, kind in covered):
                continue
            if spmv_kind == "csr_spmv":
                lowering_status = (
                    "cuSPARSE route available after i32-index and f32/f64 "
                    "operand validation")
            else:
                lowering_status = (
                    "cuSPARSE route available through the validated JDS-to-CSR "
                    "storage adapter after exact operand-role validation")
            results.append(ResidualIdiomCandidate(
                spmv_kind, loop,
                (f"row bounds loaded from {bounds_buffer}", "column-indexed gather",
                 "multiply-add row reduction"),
                lowering_status))
            covered.append((loop.span[0], loop.span[1], spmv_kind))
    return results


def _generic_computation(op: StructuredOp) -> StructuredComputation:
    parallel = LoopDomain(
        f"generic:{op.index}:parallel:{sum(it == 'parallel' for it in op.body.iterator_types)}")
    if op.reduction_count:
        reduction = LoopDomain(
            f"generic:{op.index}:reduction:{op.reduction_count}")
        computation = StructuredComputation.Reduce(
            parallel, reduction, _term_with_accesses(op))
    else:
        computation = StructuredComputation.Map(parallel, _term_with_accesses(op))
    for loop in reversed(op.loops):
        computation = StructuredComputation.For(
            LoopDomain(loop.domain_text), computation)
    return computation


def _pure_computation_without_parent_loops(
        op: StructuredOp) -> StructuredComputation:
    domain = LoopDomain(
        f"generic-chain:parallel:{sum(it == 'parallel' for it in op.body.iterator_types)}")
    if op.reduction_count:
        return StructuredComputation.Reduce(
            domain,
            LoopDomain(f"generic-chain:reduction:{op.reduction_count}"),
            _term_with_accesses(op))
    return StructuredComputation.Map(domain, _term_with_accesses(op))


def _term_with_accesses(op: StructuredOp) -> Term:
    """Replace positional scalar leaves with root+map access descriptors."""
    ast = _parse_term(_term_repr(op.term))

    def replace(node):
        if not isinstance(node, tuple):
            return node
        if node[0] == "In":
            index = int(node[1])
            if index < len(op.accesses):
                return ("Access", op.input_roots[index]
                        if index < len(op.input_roots)
                        else op.output_roots[index - len(op.input_roots)],
                        op.accesses[index])
        if node[0] == "Out":
            index = len(op.inputs) + int(node[1])
            if index < len(op.accesses):
                return ("Access", op.output_roots[int(node[1])],
                        op.accesses[index])
        return (node[0], *(replace(child) if isinstance(child, tuple)
                           else child for child in node[1:]))

    return _ast_to_term(replace(ast))


def _is_gemv_access_shape(op: StructuredOp) -> bool:
    """Check A[i,k], x[k], y[i] rather than iterator counts alone."""
    if len(op.body.indexing_maps) != 3:
        return False
    parsed = []
    for map_text in op.body.indexing_maps:
        match = re.fullmatch(
            r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>",
            map_text.strip())
        if not match:
            return False
        dims = [value.strip() for value in match.group(1).split(",")
                if value.strip()]
        results = [value.strip() for value in match.group(2).split(",")
                   if value.strip()]
        parsed.append((dims, results))
    dims = parsed[0][0]
    if len(dims) != 2 or any(item[0] != dims for item in parsed):
        return False
    parallel, reduction = dims
    return (set(parsed[0][1]) == {parallel, reduction} and
            parsed[1][1] == [reduction] and
            parsed[2][1] == [parallel])


def _has_standard_blas_operand_order(op: StructuredOp, operation: str) -> bool:
    """Require the operand order implemented by the current row-major ABI."""
    maps = [map_text.replace(" ", "") for map_text in op.body.indexing_maps]
    if operation == "gemv":
        return maps == [
            "affine_map<(d0,d1)->(d0,d1)>",
            "affine_map<(d0,d1)->(d1)>",
            "affine_map<(d0,d1)->(d0)>",
        ]
    if operation == "gemm":
        return maps == [
            "affine_map<(d0,d1,d2)->(d0,d2)>",
            "affine_map<(d0,d1,d2)->(d2,d1)>",
            "affine_map<(d0,d1,d2)->(d0,d1)>",
        ]
    return False


def _is_column_sliced_gemm(op: StructuredOp) -> bool:
    if len(op.loops) != 1 or len(op.accesses) != 3:
        return False
    normalized = [value.replace(" ", "") for value in op.accesses]
    column_map = "submap=affine_map<(d0)[s0]->(d0,s0)>"
    iv = op.loops[0].induction
    return ("submap=direct" in normalized[0] and
            column_map in normalized[1] and column_map in normalized[2] and
            f"symbols={iv}" in normalized[1] and
            f"symbols={iv}" in normalized[2])


def _is_leading_batch_sliced(op: StructuredOp, operand_ranks: tuple[int, ...]) -> bool:
    """Prove that one parent loop selects the leading mode of every operand."""
    if len(op.loops) != 1 or len(op.accesses) != len(operand_ranks):
        return False
    iv = op.loops[0].induction
    for access, rank in zip(op.accesses, operand_ranks):
        normalized = access.replace(" ", "")
        logical_dims = ",".join(f"d{i}" for i in range(rank - 1))
        physical_dims = ",".join(["s0", *[f"d{i}" for i in range(rank - 1)]])
        expected = (
            f"submap=affine_map<({logical_dims})[s0]->({physical_dims})>")
        if expected not in normalized or f"symbols={iv}" not in normalized:
            return False
    return True


def saturate_region(region: StructuredRegion,
                    iterations: int = 8) -> SaturationResult:
    """Saturate one legal region and return proof-oriented telemetry."""
    source = _generic_computation(region.operations[0])
    for op in region.operations[1:]:
        source = StructuredComputation.Sequence(source, _generic_computation(op))

    started = time.monotonic()
    diagnostics: list[str] = []
    if not region.facts.fusible:
        diagnostics.append("region rejected by imperative safety facts")
        return SaturationResult(
            region, source, None, None,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            egraph_nodes=0, iterations=0, diagnostics=diagnostics)

    # Once legality is established, erase the scheduling distinction between
    # the common parent loops and construct a pure same-domain sequence. The
    # equality itself (Sequence <-> FusedN and loop lifting) is Egglog-owned.
    sequence = _pure_computation_without_parent_loops(region.operations[0])
    for op in region.operations[1:]:
        sequence = StructuredComputation.Sequence(
            sequence, _pure_computation_without_parent_loops(op))
    for loop in reversed(region.operations[0].loops):
        sequence = StructuredComputation.For(
            LoopDomain(loop.domain_text), sequence)

    egraph = EGraph()
    egraph.register(sequence)
    # Scalar algebra and schedule algebra share one e-graph. Distributive
    # expansion is disabled here because large loop regions magnify its search
    # space; associativity, commutativity, identities and constant folding are
    # still available beneath Map/Reduce/Fused nodes.
    schedule = (structured_rules() +
                algebra_rules(include_distributivity=False)) * iterations
    report = egraph.run(schedule)
    elapsed = (time.monotonic() - started) * 1000.0
    fused: StructuredComputation | None = None
    ops = region.operations
    generic_domain = LoopDomain(
        f"generic-chain:parallel:{sum(it == 'parallel' for it in ops[0].body.iterator_types)}")
    lifted_domain = generic_domain
    for loop in reversed(ops[0].loops):
        lifted_domain = LoopDomain.Nest(LoopDomain(loop.domain_text),
                                        lifted_domain)
    if len(ops) == 1:
        if ops[0].reduction_count:
            candidate = StructuredComputation.Reduce(
                lifted_domain,
                LoopDomain(f"generic-chain:reduction:{ops[0].reduction_count}"),
                _term_with_accesses(ops[0]))
        else:
            candidate = StructuredComputation.Map(
                lifted_domain, _term_with_accesses(ops[0]))
    elif len(ops) == 2 and all(not op.reduction_count for op in ops):
        candidate = StructuredComputation.Fused2(
            lifted_domain, _term_with_accesses(ops[0]),
            _term_with_accesses(ops[1]))
    elif len(ops) == 3 and all(not op.reduction_count for op in ops):
        candidate = StructuredComputation.Fused3(
            lifted_domain, _term_with_accesses(ops[0]),
            _term_with_accesses(ops[1]), _term_with_accesses(ops[2]))
    else:
        candidate = None
        diagnostics.append("fusion extraction currently supports 2-3 stages")
    if candidate is not None:
        try:
            egraph.check(sequence == candidate)
            fused = candidate
        except Exception:
            diagnostics.append("Egglog did not prove the fused schedule")
    sizes = egraph.all_function_sizes()
    node_count = sum(size for _, size in sizes)
    extracted_kind: str | None = None
    lowering_blocker: str | None = None
    if fused is not None:
        extracted_kind = (_classify_single_op(ops[0]) if len(ops) == 1 else
                          f"fused_{len(ops)}stage_map_{ops[0].parallel_count}d")
        if extracted_kind == "dense_gemm" and not _gemm_body_equivalent(
                egraph, ops[0]):
            extracted_kind = None
            lowering_blocker = "GEMM access shape found but scalar body is not equivalent"
        # This is the factorized shape used by NPB-MG resid: two four-neighbor
        # sums in stack temporaries feed a three-dimensional linear stencil.
        # Recognizing the schedule is useful, but the existing packed cuDNN
        # ABI cannot consume the coefficient vector or its V - conv(U,W)
        # epilogue.  Keep that boundary explicit instead of reporting a false
        # executable match.
        representations = [_term_repr(op.term) for op in ops]
        if (len(ops) == 1 and len(ops[0].loops) == 1 and
                ops[0].reduction_count == 1 and
                _blas_subtract_body_equivalent(egraph, ops[0])):
            if (_is_gemm_access_shape(ops[0]) and
                    _has_standard_blas_operand_order(ops[0], "gemm") and
                    _is_leading_batch_sliced(ops[0], (3, 3, 3))):
                extracted_kind = "looped_gemm_as_batched_gemm"
                lowering_blocker = None
            elif (_is_gemv_access_shape(ops[0]) and
                    _has_standard_blas_operand_order(ops[0], "gemv") and
                    _is_leading_batch_sliced(ops[0], (3, 2, 2))):
                extracted_kind = "looped_gemv_as_batched_gemv"
                lowering_blocker = None
        if (len(ops) == 1 and ops[0].parallel_count == 2 and
                ops[0].reduction_count == 1 and
                _is_gemv_access_shape(ops[0]) and
                "Term.In(0)" in representations[0] and
                "Term.In(1)" in representations[0] and
                ("Term.Out(0)" in representations[0] or
                 (len(ops[0].body.indexing_maps) == 3 and
                  "Term.In(2)" in representations[0]))):
            if extracted_kind != "looped_gemv_as_batched_gemv":
                extracted_kind = "looped_gemv_as_gemm_schedule"
                if (_is_column_sliced_gemm(ops[0]) and
                        _gemm_body_equivalent(egraph, ops[0])):
                    extracted_kind = "looped_gemv_as_gemm"
                else:
                    lowering_blocker = (
                        "needs affine composition of parent-loop slices into "
                        "rank-2 GEMM operands before emitting cublas GEMM")
        four_sum = "Term.In(3)" in representations[0]
        if (len(ops) == 3 and ops[0].parallel_count == 3 and
                all(op.reduction_count == 0 for op in ops) and four_sum and
                representations[0] == representations[1] and
                "Term.In(9)" in representations[2]):
            extracted_kind = "factorized_linear_stencil3d"
            lowering_blocker = (
                "needs recovery of flattened rank-2 storage, packed 3x3x3 "
                "weights, and a V-convolution epilogue; current "
                "cudnnConvolution3D_ntap ABI represents pure rank-3 convolution")
    return SaturationResult(
        region, source, sequence, fused, elapsed, node_count,
        iterations, extracted_kind=extracted_kind,
        lowering_blocker=lowering_blocker,
        diagnostics=diagnostics + [str(report)])


def analyze_structured_regions(text: str, instances: Sequence,
                               bodies: Sequence[GenericBody],
                               terms: Sequence[Term | None]) -> list[SaturationResult]:
    return [saturate_region(region)
            for region in discover_regions(text, instances, bodies, terms)]


def format_residual_candidate(candidate: ResidualIdiomCandidate) -> str:
    return (
        f"kind={candidate.kind} loop={candidate.loop.domain_text} "
        f"evidence={list(candidate.evidence)} "
        f"lowering_blocker={candidate.lowering_blocker}")


def format_result(result: SaturationResult) -> str:
    ops = result.region.operations
    status = "fused" if result.fused is not None else "rejected"
    detail = (
        f"{status} bodies={[op.index for op in ops]} "
        f"parent_loops={len(ops[0].loops) if ops else 0} "
        f"lifted_parallel_dims={ops[0].parallel_count if ops else 0} "
        f"temporary_roots={list(result.region.dependence_roots)} "
        f"egglog_ms={result.elapsed_ms:.3f} nodes={result.egraph_nodes}")
    if result.extracted_kind:
        detail += f" extracted={result.extracted_kind}"
    if result.lowering_blocker:
        detail += f" lowering_blocker={result.lowering_blocker}"
    return detail
