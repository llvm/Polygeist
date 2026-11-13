"""Equality-saturation ablation switch shared by the lib_v2 matchers.

The environment variable ``EQSAT_ABLATION`` selects the rewrite rules used by
both the Linalg-level matcher (``lib_v2/match_v2.py``, via
``kernel_match.equivalent``) and the region matcher (``region_match/egg.py``):

  unset / "full"       unchanged behaviour (default)
  "none"               no rewrite rules.  The Linalg-level matcher uses exact
                       syntactic comparison; the region matcher accepts only
                       equations that are identical once the operand binding is
                       applied (congruence closure, no rewriting).
  "drop:<c>[,<c>...]"  all rules except the listed categories.

Scalar categories (the paper's Table 2): comm, assoc, ident, zero, coeff,
litfold.  Region-only categories: index, mask, sum, subdiv, perm.

``scalar_rule_groups`` restates ``kernel_match.algebra_rules`` (without the
distributivity rules, which production disables) split by category;
kernel_match.py itself is not modified.
"""
from __future__ import annotations

import functools
import os

from egglog import f64, rewrite, ruleset, vars_

SCALAR_CATS = ("comm", "assoc", "ident", "zero", "coeff", "litfold")
REGION_CATS = ("index", "mask", "sum", "subdiv", "perm")
ALL_CATS = SCALAR_CATS + REGION_CATS


def mode() -> tuple[str, frozenset]:
    v = os.environ.get("EQSAT_ABLATION", "full").strip()
    if v in ("", "full"):
        return "full", frozenset()
    if v == "none":
        return "none", frozenset()
    if v.startswith("drop:"):
        cats = frozenset(c.strip() for c in v[len("drop:"):].split(",") if c.strip())
        unknown = cats - set(ALL_CATS)
        if unknown:
            raise ValueError(f"EQSAT_ABLATION: unknown categories {sorted(unknown)}")
        return "drop", cats
    raise ValueError(f"EQSAT_ABLATION: unrecognised value {v!r}")


def scalar_rule_groups(km) -> dict[str, list]:
    """kernel_match.algebra_rules(include_distributivity=False), by category."""
    if km.numeric_model() == "strict":
        return {}
    T = km.Term
    a, b, c = vars_("a b c", T)
    (x,) = vars_("x", T)
    c1, c2 = vars_("c1 c2", f64)
    one, zero = T.Lit(1.0), T.Lit(0.0)
    return {
        "comm": [rewrite(a + b).to(b + a), rewrite(a * b).to(b * a)],
        "assoc": [rewrite(a + (b + c)).to((a + b) + c),
                  rewrite((a + b) + c).to(a + (b + c)),
                  rewrite(a * (b * c)).to((a * b) * c),
                  rewrite((a * b) * c).to(a * (b * c))],
        "ident": [rewrite(a * one).to(a), rewrite(one * a).to(a),
                  rewrite(a + zero).to(a), rewrite(zero + a).to(a)],
        "zero": [rewrite(a * zero).to(zero), rewrite(zero * a).to(zero)],
        "coeff": [rewrite(T.Lit(c1) * x + T.Lit(c2) * x).to(T.Lit(c1 + c2) * x)],
        "litfold": [rewrite(T.Lit(c1) + T.Lit(c2)).to(T.Lit(c1 + c2)),
                    rewrite(T.Lit(c1) * T.Lit(c2)).to(T.Lit(c1 * c2))],
    }


@functools.lru_cache(maxsize=None)
def _scalar_ruleset_cached(km, exclude: frozenset, numeric_model):
    rules = [r for cat, rs in scalar_rule_groups(km).items() if cat not in exclude for r in rs]
    return ruleset(*rules)


def scalar_ruleset(km, exclude: frozenset):
    return _scalar_ruleset_cached(km, frozenset(exclude), km.numeric_model())


def install_linalg_level(km) -> tuple[str, frozenset]:
    """Apply the ablation to kernel_match's scalar equivalence check."""
    m, cats = mode()
    if not hasattr(km, "_unablated_algebra_rules"):
        km._unablated_algebra_rules = km.algebra_rules
    km.algebra_rules = km._unablated_algebra_rules
    if m == "none":
        km.configure_matcher("syntactic")
    elif m == "drop":
        rs = scalar_ruleset(km, cats)
        km.algebra_rules = lambda include_distributivity=True: rs  # noqa: E731
    return m, cats
