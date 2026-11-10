"""Shared policy: may a library-spec parameter be bound to an
application literal?

Used by BOTH matchers so they accept exactly the same literal bindings:
  * region_match (element equations): a spec float parameter is unioned with
    `Term.Lit(v)` in the e-graph; an int parameter that makes the spec
    non-affine (a stride) is specialised to a constant before symbolic
    execution;
  * lib_v2/match_v2.py (Linalg bodies): a body capture of a spec parameter
    unifies with a literal (`("lit", v)`), a stride symbol binds to 1
    (`("int", 1)`).

Rules (no routine knowledge; per-routine facts come only from the manifest):
  1. the value must be finite;
  2. a float literal 0.0 is never bound: it deletes every product it multiplies,
     so the spec would drop reads whose NaN/Inf the application does not
     propagate (and the library carries explicit alpha0/beta0 specs);
  3. the value must satisfy every relational assumption the manifest entry
     states about that parameter (`alpha != 0`, `beta != 0 and beta != 1`,
     `incx > 0`, `incx == 1`, ...).
"""
from __future__ import annotations

import math
import re

_REL = re.compile(r"\b([A-Za-z_]\w*)\s*(==|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w.])")


def assumption_constraints(entry: dict) -> dict[str, list[tuple[str, float]]]:
    """{param: [(op, value), ...]} parsed from the manifest `assumptions`."""
    out: dict[str, list[tuple[str, float]]] = {}
    for a in entry.get("assumptions", []) if entry else []:
        for m in _REL.finditer(a):
            out.setdefault(m.group(1), []).append((m.group(2), float(m.group(3))))
    return out


_OPS = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def literal_ok(entry: dict, cname: str, value: float, kind: str = "float") -> tuple[bool, str]:
    """(allowed, reason).  kind is 'float' or 'int'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"{cname}: literal {value!r} is not numeric"
    if not math.isfinite(v):
        return False, f"{cname}: non-finite literal"
    if kind == "float" and v == 0.0:
        return False, f"{cname} := 0.0 would delete reads (NaN/Inf semantics differ)"
    for op, w in assumption_constraints(entry).get(cname, []):
        if not _OPS[op](v, w):
            return False, f"{cname} := {v:g} violates manifest assumption {cname} {op} {w:g}"
    return True, ""


def candidate_float_literals(app_constants) -> list[float]:
    """Literals a spec float parameter may take: the multiplicative identity
    (code that has no alpha) and every nonzero finite constant of the
    application formulas."""
    out = [1.0]
    for c in sorted({float(x) for x in app_constants}):
        if math.isfinite(c) and c != 0.0 and c not in out:
            out.append(c)
    return out
