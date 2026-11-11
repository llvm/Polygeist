"""Explain observed native/reference disagreements on the unchanged fixture inputs.

This is a source-semantics audit only, never a native timing implementation.
Native results always come from the installed ATen CUDA dispatcher.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np

OUT = Path(__file__).resolve().parent
EDGES = np.array([0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
                  .25, -.25, .75, -.75, np.inf, -np.inf, np.nan,
                  1e-20, -1e-20, 80., -80.], dtype=np.float32)


def same(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)
    if not np.isfinite(a) or not np.isfinite(b):
        return a == b
    if a == 0 and b == 0:
        return np.signbit(a) == np.signbit(b)
    return abs(float(a) - float(b)) <= 1e-5 + 1e-4 * abs(float(b))


def input_value(i, a, edge):
    if edge:
        return EDGES[(i // len(EDGES) ** a) % len(EDGES)]
    return np.float32((((i + a * 613) * 104729) % 8192) / 1024 - 4)


rows = []
with np.errstate(all='ignore'):
    for name in ['softshrink', 'hardswish_backward']:
        for edge in [False, True]:
            counts = Counter()
            for i in range(4096):
                x = input_value(i, 0 if name == 'softshrink' else 1, edge)
                if name == 'softshrink':
                    lambd = np.float32(.25)
                    raised = (x - lambd if x > lambd else
                              x + lambd if x < -lambd else x * np.float32(0))
                    native = (x - lambd if x > lambd else
                              x + lambd if x < -lambd else np.float32(0))
                    cause = 'NaN versus zero' if np.isnan(x) else 'negative zero versus positive zero'
                else:
                    grad = input_value(i, 0, edge)
                    middle = np.float32(grad * np.float32(np.float32(x / np.float32(3)) + np.float32(.5)))
                    raised = np.float32(0) if x <= -3 else middle if x < 3 else grad
                    native = np.float32(0) if x < -3 else middle if x <= 3 else grad
                    cause = 'self=-3 boundary' if x == -3 else 'self=+3 boundary' if x == 3 else 'other'
                if not same(raised, native):
                    counts[cause] += 1
            observed = {('softshrink', False): 122, ('softshrink', True): 862,
                        ('hardswish_backward', False): 1, ('hardswish_backward', True): 317}[name, edge]
            assert sum(counts.values()) == observed, (name, edge, counts, observed)
            rows.append(dict(name=name, edge=edge, parameter_case=0, count=4096,
                             observed_native_mismatches_per_residency_mode=observed,
                             source_semantics_predicted_mismatches=dict(counts)))
(OUT / 'disagreement-audit.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
